//! Read-only access to the few registry keys the hardware probe needs.
//!
//! The display-adapter and compute-accelerator device classes, the Windows version key and the WSL
//! distribution list have no other unprivileged, shell-free API. Everything here opens keys with
//! `KEY_READ` only and never writes. Each `unsafe` block is one FFI call with a `SAFETY` comment;
//! the key handle is closed by [`Key`]'s `Drop`, so no path leaks it.

use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::{ERROR_MORE_DATA, ERROR_NO_MORE_ITEMS, ERROR_SUCCESS};
use windows_sys::Win32::System::Registry::{
    RegCloseKey, RegEnumKeyExW, RegGetValueW, RegOpenKeyExW, HKEY, KEY_READ, RRF_RT_REG_BINARY,
    RRF_RT_REG_DWORD, RRF_RT_REG_QWORD, RRF_RT_REG_SZ,
};

pub(crate) use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};

/// Longest value Cerebro reads (a driver description or version string). Registry strings this
/// long are not real hardware data; refusing them bounds the allocation.
const MAX_VALUE_BYTES: u32 = 64 * 1024;

/// Upper bound on subkeys enumerated under one key. A device class has a few dozen entries; the
/// bound only stops a corrupted hive from looping the probe.
const MAX_SUBKEYS: u32 = 4096;

/// An open, read-only registry key, closed on drop.
pub(crate) struct Key(HKEY);

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: `self.0` was returned by a successful `RegOpenKeyExW` and is closed exactly once,
        // here. The return value is ignored: nothing useful can be done about a failed close.
        unsafe {
            RegCloseKey(self.0);
        }
    }
}

fn wide(text: &str) -> Vec<u16> {
    std::ffi::OsStr::new(text)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

impl Key {
    /// Open `path` under a predefined root (or under another open key) for reading. `Err` carries
    /// the Win32 status (2 = not found, 5 = access denied).
    pub(crate) fn open(root: HKEY, path: &str) -> Result<Self, u32> {
        let path = wide(path);
        let mut out: HKEY = std::ptr::null_mut();
        // SAFETY: `path` is a NUL-terminated UTF-16 buffer that outlives the call, and `out` is a
        // live local the API writes the opened handle into on success. `root` is a predefined
        // root or a handle owned by a live `Key`.
        let status = unsafe { RegOpenKeyExW(root, path.as_ptr(), 0, KEY_READ, &mut out) };
        if status == ERROR_SUCCESS {
            Ok(Self(out))
        } else {
            Err(status)
        }
    }

    /// Open the subkey `name` of this key.
    pub(crate) fn subkey(&self, name: &str) -> Result<Self, u32> {
        Self::open(self.0, name)
    }

    /// Names of this key's direct subkeys, in enumeration order. A subkey name that is not valid
    /// UTF-16 is skipped (device-class and Lxss subkeys are ASCII by construction).
    pub(crate) fn subkey_names(&self) -> Result<Vec<String>, u32> {
        let mut names = Vec::new();
        // 256 UTF-16 units is the documented maximum key-name length (255 plus NUL).
        let mut buf = [0u16; 256];
        for index in 0..MAX_SUBKEYS {
            let mut len = buf.len() as u32;
            // SAFETY: `buf` holds `len` u16s and `len` is passed by pointer as its capacity, so the
            // API cannot write past it; it writes the name length (without NUL) back into `len`.
            // The reserved, class and time pointers are null, which the API allows.
            let status = unsafe {
                RegEnumKeyExW(
                    self.0,
                    index,
                    buf.as_mut_ptr(),
                    &mut len,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if status == ERROR_NO_MORE_ITEMS {
                return Ok(names);
            }
            if status != ERROR_SUCCESS {
                return Err(status);
            }
            if let Some(name) = buf.get(..len as usize) {
                if let Ok(name) = String::from_utf16(name) {
                    names.push(name);
                }
            }
        }
        Ok(names)
    }

    /// Read raw value bytes with the given type filter. `Ok(None)` when the value does not exist
    /// or has a different type.
    fn raw(&self, value: &str, flags: u32) -> Result<Option<Vec<u8>>, u32> {
        let name = wide(value);
        let mut size: u32 = 0;
        // SAFETY: sizing call: a null data pointer with a live `size` asks for the byte count only.
        // `name` is NUL-terminated and outlives the call; the subkey pointer is null (this key).
        let status = unsafe {
            RegGetValueW(
                self.0,
                std::ptr::null(),
                name.as_ptr(),
                flags,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        };
        // 2 = not found; 13 (ERROR_INVALID_DATA) / 1630 (ERROR_UNSUPPORTED_TYPE) = wrong type.
        if status == 2 || status == 13 || status == 1630 {
            return Ok(None);
        }
        if status != ERROR_SUCCESS && status != ERROR_MORE_DATA {
            return Err(status);
        }
        if size > MAX_VALUE_BYTES {
            return Err(ERROR_MORE_DATA);
        }
        let mut data = vec![0u8; size as usize];
        let mut written = size;
        // SAFETY: `data` is a live buffer of exactly `written` bytes and `written` is passed as its
        // capacity, so the API cannot write past it; it stores the byte count actually written.
        let status = unsafe {
            RegGetValueW(
                self.0,
                std::ptr::null(),
                name.as_ptr(),
                flags,
                std::ptr::null_mut(),
                data.as_mut_ptr().cast(),
                &mut written,
            )
        };
        if status == 2 || status == 13 || status == 1630 {
            return Ok(None);
        }
        if status != ERROR_SUCCESS {
            // ERROR_MORE_DATA here means the value grew between the two calls; report, never retry
            // unboundedly.
            return Err(status);
        }
        data.truncate(written as usize);
        Ok(Some(data))
    }

    /// A `REG_SZ` value as text, NUL terminator and surrounding whitespace removed. `Ok(None)`
    /// when absent, of another type, empty, or not valid UTF-16.
    pub(crate) fn string(&self, value: &str) -> Result<Option<String>, u32> {
        let Some(bytes) = self.raw(value, RRF_RT_REG_SZ)? else {
            return Ok(None);
        };
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .filter_map(|pair| match pair {
                [lo, hi] => Some(u16::from_le_bytes([*lo, *hi])),
                _ => None,
            })
            .take_while(|&u| u != 0)
            .collect();
        Ok(String::from_utf16(&units)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()))
    }

    /// An integer stored as `REG_DWORD`, `REG_QWORD`, or a 4- or 8-byte `REG_BINARY` (display
    /// drivers use all three for memory sizes). `Ok(None)` when absent or of another shape.
    pub(crate) fn integer(&self, value: &str) -> Result<Option<u64>, u32> {
        let Some(bytes) =
            self.raw(value, RRF_RT_REG_DWORD | RRF_RT_REG_QWORD | RRF_RT_REG_BINARY)?
        else {
            return Ok(None);
        };
        Ok(le_integer(&bytes))
    }
}

/// A little-endian 4- or 8-byte integer; any other length is not an integer value.
fn le_integer(bytes: &[u8]) -> Option<u64> {
    match bytes.len() {
        4 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(bytes);
            Some(u64::from(u32::from_le_bytes(b)))
        }
        8 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(bytes);
            Some(u64::from_le_bytes(b))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_are_read_only_from_four_or_eight_bytes() {
        assert_eq!(le_integer(&[0, 0, 0, 2]), Some(0x0200_0000));
        assert_eq!(le_integer(&8_589_934_592u64.to_le_bytes()), Some(8_589_934_592));
        assert_eq!(le_integer(&[1, 2, 3]), None);
        assert_eq!(le_integer(&[]), None);
    }

    #[test]
    fn the_windows_version_key_is_readable() {
        // Present on every Windows install and readable without privilege.
        let key = Key::open(HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            .unwrap();
        let build = key.string("CurrentBuildNumber").unwrap().unwrap();
        assert!(build.parse::<u32>().is_ok(), "{build:?}");
        assert!(key.integer("CurrentMajorVersionNumber").unwrap().is_some());
        // A string value is not an integer, and a missing value is None, not an error.
        assert_eq!(key.integer("CurrentBuildNumber").unwrap(), None);
        assert_eq!(key.string("CerebroValueThatDoesNotExist").unwrap(), None);
        assert!(matches!(key.subkey("CerebroKeyThatDoesNotExist"), Err(2)));
    }
}
