//! Whether the machine is on mains power or battery, as a measurement condition.
//!
//! A laptop on battery runs its CPU and GPU under a lower power limit, and Windows' battery saver
//! throttles further, so the same configuration measures slower unplugged. A throughput number that
//! does not say which state it was taken in cannot be compared with another. One
//! `GetSystemPowerStatus` call answers it; it reads a cached kernel structure, costs microseconds,
//! and needs no privilege.
//!
//! The `unsafe` surface is two FFI calls, each carrying a `SAFETY` comment.

use crate::error::ProbeError;

/// Where the machine is drawing power from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PowerSource {
    /// Mains power (the AC adapter is connected).
    Ac,
    /// Battery (the AC adapter is not connected).
    Battery,
    /// The OS reported the line status as unknown (`ACLineStatus == 255`), which desktops without
    /// a battery driver sometimes do.
    Unknown,
}

/// The machine's power state at one moment. Perishable: sample it next to the measurement it
/// describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PowerState {
    /// Mains or battery.
    pub source: PowerSource,
    /// Whether Windows' battery saver is on (it lowers CPU clocks and background activity).
    pub battery_saver: bool,
    /// Remaining battery charge in percent, `None` when there is no battery or it is unknown.
    pub battery_percent: Option<u8>,
}

impl PowerState {
    /// Decode the raw `SYSTEM_POWER_STATUS` fields. Split out so every mapping is testable
    /// without the machine being in that state.
    pub fn from_raw(ac_line_status: u8, battery_life_percent: u8, system_status_flag: u8) -> Self {
        let source = match ac_line_status {
            0 => PowerSource::Battery,
            1 => PowerSource::Ac,
            _ => PowerSource::Unknown,
        };
        Self {
            source,
            battery_saver: system_status_flag == 1,
            battery_percent: (battery_life_percent <= 100).then_some(battery_life_percent),
        }
    }
}

/// Sample the current power state.
///
/// # Platform
///
/// Windows only; every other platform returns [`ProbeError::UnsupportedPlatform`].
pub fn probe() -> Result<PowerState, ProbeError> {
    #[cfg(windows)]
    {
        win::query()
    }
    #[cfg(not(windows))]
    {
        Err(ProbeError::UnsupportedPlatform)
    }
}

#[cfg(windows)]
mod win {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};

    use super::PowerState;
    use crate::error::ProbeError;

    pub(super) fn query() -> Result<PowerState, ProbeError> {
        let mut status = SYSTEM_POWER_STATUS {
            ACLineStatus: 0,
            BatteryFlag: 0,
            BatteryLifePercent: 0,
            SystemStatusFlag: 0,
            BatteryLifeTime: 0,
            BatteryFullLifeTime: 0,
        };
        // SAFETY: `status` is a live, properly aligned local of the exact struct type the API
        // writes; the call writes only within it and keeps no pointer after returning.
        if unsafe { GetSystemPowerStatus(&mut status) } == 0 {
            // SAFETY: reads the calling thread's last-error value immediately after the failed call.
            let code = unsafe { GetLastError() };
            return Err(ProbeError::Win32 {
                call: "GetSystemPowerStatus",
                code,
            });
        }
        Ok(PowerState::from_raw(
            status.ACLineStatus,
            status.BatteryLifePercent,
            status.SystemStatusFlag,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_fields_decode_to_the_documented_states() {
        assert_eq!(
            PowerState::from_raw(1, 80, 0),
            PowerState {
                source: PowerSource::Ac,
                battery_saver: false,
                battery_percent: Some(80)
            }
        );
        assert_eq!(
            PowerState::from_raw(0, 42, 1),
            PowerState {
                source: PowerSource::Battery,
                battery_saver: true,
                battery_percent: Some(42)
            }
        );
        // 255 is "unknown" for both the line status and the charge.
        assert_eq!(
            PowerState::from_raw(255, 255, 0),
            PowerState {
                source: PowerSource::Unknown,
                battery_saver: false,
                battery_percent: None
            }
        );
        assert_eq!(PowerState::from_raw(1, 101, 0).battery_percent, None);
    }

    #[cfg(windows)]
    #[test]
    fn the_live_query_succeeds() {
        // Which state this machine is in is a fact about the environment; this proves only that
        // the FFI call itself is sound and does not report failure.
        let _ = probe().expect("GetSystemPowerStatus must succeed");
    }
}
