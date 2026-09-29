//! Runtime ground truth about the machine Cerebro is tuning.
//!
//! Cerebro picks a llama.cpp configuration by measurement, and a search is only as good as the
//! facts it starts from. This crate supplies those facts, and it supplies them by *asking the
//! running system*, never by assuming. Two consequences run through everything here:
//!
//! * **Nothing is hardcoded.** An affinity mask copied from a development box would pin inference
//!   threads onto logical processors that may not exist on the user's CPU, and Windows accepts that
//!   silently. Every mask in [`cpu`] is derived at runtime.
//! * **A wrong answer is worse than no answer.** A plausible-but-false topology poisons every
//!   measurement taken afterwards, so probes that cannot establish the truth return a
//!   [`ProbeError`] rather than a best guess. That includes cases where the underlying OS call
//!   reported success but the data it wrote cannot be right.
//!
//! `unsafe` is present and unavoidable: efficiency class and NUMA topology have no portable API,
//! and shelling out to PowerShell for them would be both slower and far easier to spoof. Each block
//! is a single FFI call carrying a `SAFETY` comment, and no path reachable from a public function
//! panics, unwraps, or indexes a slice unchecked — malformed kernel data becomes a typed error.

#![deny(missing_docs)]

pub mod backend;
pub mod child_env;
pub mod cpu;
pub mod error;
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures;
pub mod gpu;
pub mod token;

pub use backend::{
    AddressableDevice, AuditedObject, BackendIdentity, BackendProbe, DaclAudit, GrantVia,
    UnsafeDacl, WritableGrant,
};
pub use cpu::{CpuTopology, PhysicalCore};
pub use error::ProbeError;
pub use gpu::{GpuInventory, PhysicalGpu};
