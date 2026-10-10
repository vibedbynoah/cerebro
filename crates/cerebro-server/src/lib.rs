//! Cerebro's loopback gateway: one supervised `llama-server` behind an authenticated,
//! OpenAI-compatible API on `127.0.0.1`.
//!
//! - [`serve()`] launches llama-server through `cerebro-supervisor` (preflight before every launch
//!   and restart, the user's protections from settings, tuned or default load flags) and starts
//!   the gateway in front of it.
//! - [`gateway`] is the HTTP surface: `/v1/models`, `/v1/chat/completions` (SSE or not),
//!   `/v1/completions`, `/v1/embeddings`, `/cerebro/health`, `/cerebro/status`.
//! - [`guard`] runs before every route: Host must name the gateway on loopback (DNS-rebinding
//!   defence), Origin must be allowlisted (none by default, so CORS is closed), and a bearer token
//!   is required everywhere, reads included. Each of the three is on by default and can be
//!   switched off in settings ([`GatewayProtections`]); one that is off is listed once in status.
//! - [`token`]: the gateway token (256 bits from the OS CSPRNG, kept in the DPAPI secret store,
//!   compared in constant time) and llama-server's separate per-launch key.
//! - [`effort`] applies the effort and verbosity dials to a chat request: instructions appended to
//!   the system prompt, the thinking budget mapped to llama-server's reasoning budget with the
//!   answer budget reserved apart from it.
//! - [`passes`] runs XHigh's self-review and Max's self-review plus independent check before the
//!   answer is returned, timing each pass for the reply's `cerebro` field.
//! - [`status`] reports the model, the last measured speeds and VRAM.
//! - [`tuned`] binds a tuned recommendation to the model, backend build, machine and flags its
//!   measurement record describes, before `serve` uses it.
//!
//! Nothing in this crate logs; prompts, completions and tokens are never written anywhere.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod effort;
pub mod error;
pub mod gateway;
pub mod guard;
pub mod passes;
pub mod serve;
pub mod status;
pub mod token;
pub mod tuned;
pub mod upstream;

pub use effort::{Dials, Plan};
pub use error::{ApiError, ServerError};
pub use gateway::{Gateway, GatewayConfig, GatewayHandle, GatewayListener, GatewayProtections, LaunchSink, Timeouts};
pub use serve::{serve, LoadFlags, ServeRequest, Serving};
pub use status::{GpuMemory, Telemetry, VramSource};
pub use token::BearerToken;
pub use upstream::{TargetSource, Upstream, UpstreamTarget};
