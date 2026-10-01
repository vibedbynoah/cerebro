//! Which device-class instances belong to devices that are present right now (CfgMgr32).
//!
//! Windows keeps a `Class\{guid}\NNNN` driver key for every device that ever had a driver
//! installed: a GPU that was swapped out, a disconnected eGPU, an iGPU disabled in firmware, an old
//! NPU driver. Reading the class keys alone would report hardware that is not in the machine. The
//! configuration manager knows which device nodes are present; each present node's `Driver`
//! property names its class key (`{4d36e968-...}\0001`), so the probe keeps exactly those.
//!
//! Each `unsafe` block is one FFI call with a `SAFETY` comment. Return codes are `CONFIGRET`
//! values (compared against `CR_SUCCESS`), not Win32 error codes.

use std::collections::BTreeSet;
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_DevNode_Registry_PropertyW, CM_Get_Device_ID_ListW, CM_Get_Device_ID_List_SizeW,
    CM_Locate_DevNodeW, CM_DRP_DRIVER, CM_GETIDLIST_FILTER_CLASS, CM_GETIDLIST_FILTER_PRESENT,
    CM_LOCATE_DEVNODE_NORMAL, CR_SUCCESS,
};

/// Upper bound on the device-ID list, in UTF-16 units (2 MiB). A class lists a few dozen IDs of
/// under 200 characters each; the bound only stops a corrupted answer from sizing the allocation.
const MAX_LIST_UNITS: u32 = 1 << 20;

/// Longest `Driver` property read, in UTF-16 units. The value is `{guid}\NNNN`, 43 characters.
const DRIVER_PROPERTY_UNITS: usize = 256;

fn wide(text: &str) -> Vec<u16> {
    std::ffi::OsStr::new(text)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

/// The `NNNN` instance names under `Class\{class}` whose device is present now. `Err` when the
/// configuration manager cannot answer; the caller then reports the device list as unknown rather
/// than falling back to the unfiltered (possibly stale) class keys.
pub(super) fn present_driver_keys(class: &str) -> Result<BTreeSet<String>, String> {
    let filter = wide(class);
    let flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    let mut len: u32 = 0;
    // SAFETY: `len` is a live out-parameter the call writes one u32 into; `filter` is a
    // NUL-terminated UTF-16 buffer that outlives the call.
    let cr = unsafe { CM_Get_Device_ID_List_SizeW(&mut len, filter.as_ptr(), flags) };
    if cr != CR_SUCCESS {
        return Err(format!("present devices of class {class} could not be listed (CONFIGRET {cr})"));
    }
    if len > MAX_LIST_UNITS {
        return Err(format!("device list of class {class} is implausibly long ({len} units)"));
    }
    // At least the list's own double-NUL terminator.
    let mut buf = vec![0u16; len.max(2) as usize];
    // SAFETY: `buf` holds exactly `buf.len()` u16s, and that length is passed as its capacity, so
    // the call cannot write past it. `filter` is NUL-terminated and outlives the call.
    let cr = unsafe {
        CM_Get_Device_ID_ListW(filter.as_ptr(), buf.as_mut_ptr(), buf.len() as u32, flags)
    };
    if cr != CR_SUCCESS {
        // CR_BUFFER_SMALL here means a device arrived between the two calls: reported, never
        // retried in a loop.
        return Err(format!("present devices of class {class} could not be listed (CONFIGRET {cr})"));
    }
    Ok(buf
        .split(|&c| c == 0)
        .filter(|id| !id.is_empty())
        .filter_map(devnode_driver)
        .filter_map(|driver| instance_of_class(&driver, class))
        .collect())
}

/// The `Driver` property (`{class-guid}\NNNN`) of the device with this instance ID, or `None` when
/// the node cannot be located or has no driver key (no driver installed).
fn devnode_driver(instance_id: &[u16]) -> Option<String> {
    let id: Vec<u16> = instance_id.iter().copied().chain(std::iter::once(0)).collect();
    let mut node: u32 = 0;
    // SAFETY: `node` is a live out-parameter; `id` is a NUL-terminated UTF-16 buffer that outlives
    // the call.
    let cr = unsafe { CM_Locate_DevNodeW(&mut node, id.as_ptr(), CM_LOCATE_DEVNODE_NORMAL) };
    if cr != CR_SUCCESS {
        return None;
    }
    let mut buf = [0u16; DRIVER_PROPERTY_UNITS];
    let mut bytes = (buf.len() * 2) as u32;
    // SAFETY: `buf` is a live array of `bytes` bytes and `bytes` is passed by pointer as its
    // capacity, so the call cannot write past it; it writes the byte count back. The data-type
    // out-pointer is null, which the API allows.
    let cr = unsafe {
        CM_Get_DevNode_Registry_PropertyW(
            node,
            CM_DRP_DRIVER,
            std::ptr::null_mut(),
            buf.as_mut_ptr().cast(),
            &mut bytes,
            0,
        )
    };
    if cr != CR_SUCCESS {
        return None;
    }
    let units = buf.get(..(bytes as usize / 2).min(buf.len()))?;
    let text: Vec<u16> = units.iter().copied().take_while(|&u| u != 0).collect();
    String::from_utf16(&text).ok()
}

/// `NNNN` from a `Driver` property when its class GUID is `class` (case-insensitive). A driver key
/// of another class, or one that is not `{guid}\NNNN`, is not an instance of this class.
pub(super) fn instance_of_class(driver: &str, class: &str) -> Option<String> {
    let (guid, instance) = driver.rsplit_once('\\')?;
    (guid.eq_ignore_ascii_case(class)
        && instance.len() == 4
        && instance.bytes().all(|b| b.is_ascii_digit()))
    .then(|| instance.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DISPLAY: &str = "{4d36e968-e325-11ce-bfc1-08002be10318}";

    #[test]
    fn driver_keys_match_on_class_guid_and_instance_number() {
        assert_eq!(
            instance_of_class("{4D36E968-E325-11CE-BFC1-08002BE10318}\\0001", DISPLAY),
            Some("0001".into())
        );
        // Another class's 0001 is not a display adapter's 0001.
        assert_eq!(
            instance_of_class("{f01a9d53-3ff6-48d2-9f97-c8a7004be10c}\\0001", DISPLAY),
            None
        );
        assert_eq!(instance_of_class("0001", DISPLAY), None);
        assert_eq!(instance_of_class(&format!("{DISPLAY}\\Properties"), DISPLAY), None);
    }

    /// Every machine with a display has at least one present display adapter (the basic display
    /// driver at worst), and each key named is a real `NNNN` instance.
    #[test]
    fn the_present_display_adapters_are_listed() {
        let keys = present_driver_keys(DISPLAY).unwrap();
        assert!(!keys.is_empty());
        assert!(keys.iter().all(|k| k.len() == 4 && k.bytes().all(|b| b.is_ascii_digit())), "{keys:?}");
    }
}
