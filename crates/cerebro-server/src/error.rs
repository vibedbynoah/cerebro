//! Errors: [`ServerError`] for starting and stopping the gateway, [`ApiError`] for one HTTP
//! request.
//!
//! Neither type ever carries a prompt, a completion, a token or a key: messages name what went
//! wrong and where, never the content involved.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror::Error;

/// Everything that can stop the gateway from starting, or a supervised launch from being set up.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServerError {
    /// The OS random number generator refused to produce bytes.
    #[error("the OS random number generator failed: {0}")]
    Random(String),

    /// The DPAPI secret store failed (open, read, write or decrypt).
    #[error("secret store: {0}")]
    Secrets(#[from] cerebro_config::SecretsError),

    /// The stored gateway token is not the 32 bytes this crate writes. It is never repaired
    /// silently: a client configured with the old token would stop working without explanation.
    #[error(
        "the stored gateway token is {found} bytes, not {expected}; rotate it to issue a new one"
    )]
    TokenCorrupt {
        /// Bytes found in the store.
        found: usize,
        /// Bytes a token must have.
        expected: usize,
    },

    /// The gateway could not bind its loopback listener.
    #[error("could not bind the gateway on {addr}: {cause}")]
    Bind {
        /// The address that was asked for.
        addr: String,
        /// The OS error.
        cause: String,
    },

    /// A directory Cerebro needs for a launch (keys, temp, driver cache, logs) could not be
    /// created or swept.
    #[error("could not prepare {path}: {cause}")]
    Io {
        /// The path involved.
        path: String,
        /// The OS error.
        cause: String,
    },

    /// The load flags handed to [`crate::serve`] include a flag preflight owns (host, port, key
    /// file, model, web UI, slots, CORS); they are set by the supervisor and may not be overridden.
    #[error("load flag {0} is owned by preflight and may not be passed as a load flag")]
    ReservedFlag(&'static str),

    /// The results store holding the tuned configuration's measurement failed to load or verify.
    #[error("results store: {0}")]
    Store(#[from] cerebro_store::StoreError),

    /// The tuned configuration names a measurement record the verified store does not hold.
    #[error("the tuned configuration names record {index}, but the verified store holds {len}")]
    TunedRecordMissing {
        /// The record index that was named.
        index: usize,
        /// How many verified records the store holds.
        len: usize,
    },

    /// The tuned configuration's record does not describe what is about to be served: another
    /// model file, backend build or machine profile, or flags other than the ones it measured.
    /// Serving it would label an unmeasured configuration as measured.
    #[error("the tuned configuration does not apply: {what}")]
    TunedMismatch {
        /// The first difference found.
        what: String,
    },

    /// The tuned configuration could not be checked against what is about to be served (the
    /// model could not be hashed, or the machine could not be probed), so it is not used.
    #[error("the tuned configuration could not be verified: {what}")]
    TunedUnverifiable {
        /// What could not be read.
        what: String,
    },

    /// The supervisor refused the launch configuration or failed to stop.
    #[error("supervisor: {0}")]
    Supervisor(#[from] cerebro_supervisor::LifecycleError),

    /// The model is not inside a configured model folder, or settings refused something else.
    #[error("settings: {0}")]
    Config(#[from] cerebro_config::ConfigError),

    /// The gateway configuration is not usable (an origin that is not an origin, a zero limit).
    #[error("invalid gateway configuration: {0}")]
    InvalidConfig(String),
}

/// One failed request, rendered as an OpenAI-style error body:
/// `{"error":{"message":..,"type":..,"code":<status>}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    /// HTTP status sent to the client.
    pub status: StatusCode,
    /// Stable machine-readable kind, such as `"unauthorized"` or `"upstream_unavailable"`.
    pub kind: &'static str,
    /// Human-readable explanation. Never contains request or response content.
    pub message: String,
}

impl ApiError {
    /// Build an error from its parts.
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into() }
    }

    /// 400: the request body is not what the route accepts.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    /// 502: the model server answered with something the gateway could not use.
    pub fn bad_upstream(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "upstream_error", message)
    }

    /// 503: the model server is not ready (starting, restarting, stopped or failed).
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "upstream_unavailable", message)
    }

    /// 504: the model server did not answer within the configured timeout.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, "timeout", message)
    }

    /// The JSON body this error is rendered as.
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "code": self.status.as_u16(),
            }
        })
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status.as_u16(), self.kind, self.message)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, json_body(&self.body())).into_response();
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

/// Serialize a JSON value for a response body. `serde_json::Value` always serializes, so the
/// fallback is unreachable in practice; it exists so no path here can panic.
pub(crate) fn json_body(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_renders_openai_shape_and_bearer_challenge() {
        let response = ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "missing token")
            .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let body = ApiError::bad_request("x").body();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], 400);
        assert_eq!(body["error"]["message"], "x");
    }
}
