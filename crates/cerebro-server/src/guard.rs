//! The checks every request passes before any route runs, in this order (each of the Host,
//! Origin and token checks can be switched off in settings, `protections.gateway_*`; see
//! [`crate::gateway::GatewayProtections`]):
//!
//! 1. **Host** must be `127.0.0.1:<port>` or `localhost:<port>` (`[::1]:<port>` when bound to
//!    IPv6 loopback), with the gateway's own port. A DNS-rebinding page reaches a loopback port
//!    under its own host name, so it fails here before anything else is looked at.
//! 2. **Origin**, when present, must be on the allowlist (empty by default; the desktop app's
//!    origin when configured). Browsers send it on every cross-origin request, so a web page that
//!    is not the app is refused even if it somehow holds the token.
//! 3. A **CORS preflight** (`OPTIONS` from an allowed origin) is answered here with the allowed
//!    origin echoed and no data. CORS is closed by default because the allowlist is empty.
//! 4. **Bearer token** on every other request, reads included, compared in constant time.
//! 5. **Body size**: a declared `Content-Length` over the limit is refused before the body is read.
//!
//! A switched-off check is skipped and nothing else changes: with the Host check off the listener
//! still binds loopback and still refuses a peer that is not local; with the Origin check off any
//! origin is accepted and echoed in the CORS headers (never `*`); with the token check off a
//! request without a token is served.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::{ApiError, ServerError};
use crate::gateway::AppState;

/// Whether `host` (a `Host` header value) names this gateway: `127.0.0.1:<port>` or
/// `localhost:<port>`, plus `[::1]:<port>` when `ipv6` is set. The port must be present.
pub fn host_allowed(host: &[u8], port: u16, ipv6: bool) -> bool {
    let Ok(host) = std::str::from_utf8(host) else {
        return false;
    };
    let Some((name, host_port)) = host.rsplit_once(':') else {
        return false;
    };
    // The port exactly as `u16`'s Display writes it (digits only, no sign, no leading zero),
    // compared without allocating on every request.
    let canonical = host_port.bytes().all(|b| b.is_ascii_digit()) && (host_port == "0" || !host_port.starts_with('0'));
    if !canonical || host_port.parse::<u16>() != Ok(port) {
        return false;
    }
    name == "127.0.0.1" || name.eq_ignore_ascii_case("localhost") || (ipv6 && name == "[::1]")
}

/// Normalize one configured origin: lowercase, no trailing slash, `scheme://host[:port]` with no
/// path, query, userinfo, wildcard or whitespace.
///
/// # Errors
/// [`ServerError::InvalidConfig`] for anything that is not a single concrete origin (`*` and `null`
/// included: either would open the gateway to every page).
///
/// The rule is `cerebro-config`'s ([`cerebro_config::schema::normalize_origin`]), so a settings
/// file that validates is a gateway configuration that binds.
pub fn normalize_origin(origin: &str) -> Result<String, ServerError> {
    cerebro_config::schema::normalize_origin(origin)
        .ok_or_else(|| ServerError::InvalidConfig(format!("{origin:?} is not an origin (scheme://host[:port])")))
}

/// What the Origin check decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginCheck {
    /// No `Origin` header: not a browser cross-origin request.
    Absent,
    /// An allowlisted origin, echoed back in CORS headers.
    Allowed(HeaderValue),
    /// An origin that is not on the allowlist.
    Refused,
}

/// Check the `Origin` header against the (already normalized) allowlist.
pub fn check_origin(headers: &HeaderMap, allowed: &[String]) -> OriginCheck {
    let Some(origin) = headers.get(header::ORIGIN) else {
        return OriginCheck::Absent;
    };
    let Ok(text) = origin.to_str() else {
        return OriginCheck::Refused;
    };
    let text = text.trim_end_matches('/');
    if allowed.iter().any(|a| a.eq_ignore_ascii_case(text)) {
        OriginCheck::Allowed(origin.clone())
    } else {
        OriginCheck::Refused
    }
}

/// Longest `Access-Control-Request-Headers` echoed back to an allowed origin.
const MAX_REQUEST_HEADERS_LEN: usize = 1024;

/// The headers a preflight from an allowed origin may send: the ones it asked for
/// (`Access-Control-Request-Headers`, such as the `x-stainless-*` headers browser OpenAI SDKs add),
/// when that is a short list of header names, else `authorization, content-type`. This widens
/// nothing: the token is still required on the actual request.
fn allowed_request_headers(headers: &HeaderMap) -> HeaderValue {
    headers
        .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
        .filter(|v| {
            let bytes = v.as_bytes();
            !bytes.is_empty()
                && bytes.len() <= MAX_REQUEST_HEADERS_LEN
                && bytes.iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b',' | b' '))
        })
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("authorization, content-type"))
}

/// Add the CORS response headers for an allowed origin.
fn add_cors(headers: &mut HeaderMap, origin: HeaderValue) {
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
}

/// The middleware: Host, Origin, preflight, token, declared body size. See the module docs.
pub(crate) async fn guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let protections = state.config.protections;
    let host_ok = !protections.check_host
        || headers
            .get(header::HOST)
            .is_some_and(|h| host_allowed(h.as_bytes(), state.port, state.ipv6));
    if !host_ok {
        return ApiError::new(StatusCode::FORBIDDEN, "host_not_allowed", "the Host header must name this gateway on loopback")
            .into_response();
    }

    let origin = if protections.check_origin {
        match check_origin(headers, &state.config.allowed_origins) {
            OriginCheck::Refused => {
                return ApiError::new(StatusCode::FORBIDDEN, "origin_not_allowed", "this origin is not allowed").into_response();
            }
            OriginCheck::Allowed(o) => Some(o),
            OriginCheck::Absent => None,
        }
    } else {
        // The user switched the allowlist off: every origin is echoed as it came, never `*`.
        headers.get(header::ORIGIN).cloned()
    };

    if request.method() == Method::OPTIONS {
        if let Some(origin) = origin {
            let mut response = StatusCode::NO_CONTENT.into_response();
            let h = response.headers_mut();
            add_cors(h, origin);
            h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
            h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, allowed_request_headers(headers));
            h.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("600"));
            return response;
        }
    }

    // The read guard lives only for the compare (no await while it is held); a poisoned lock
    // refuses.
    let authorized = !protections.require_token
        || headers
            .get(header::AUTHORIZATION)
            .is_some_and(|v| state.token.read().is_ok_and(|token| token.verify_authorization(v.as_bytes())));
    if !authorized {
        let mut response = ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "a valid bearer token is required").into_response();
        if let Some(origin) = origin {
            add_cors(response.headers_mut(), origin);
        }
        return response;
    }

    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|n| n > state.config.max_body_bytes as u64) {
        return payload_too_large(state.config.max_body_bytes).into_response();
    }

    let mut response = next.run(request).await;
    if let Some(origin) = origin {
        add_cors(response.headers_mut(), origin);
    }
    response
}

/// 413 with the limit in the message.
pub(crate) fn payload_too_large(limit: usize) -> ApiError {
    ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", format!("the request body is over the {limit}-byte limit"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_must_be_loopback_name_with_our_port() {
        assert!(host_allowed(b"127.0.0.1:8127", 8127, false));
        assert!(host_allowed(b"localhost:8127", 8127, false));
        assert!(host_allowed(b"LocalHost:8127", 8127, false));
        assert!(host_allowed(b"[::1]:8127", 8127, true));
        for bad in [
            &b"[::1]:8127"[..],
            b"127.0.0.1",
            b"localhost",
            b"127.0.0.1:8128",
            b"evil.example:8127",
            b"127.0.0.1.evil.example:8127",
            b"localhost.:8127",
            b"0.0.0.0:8127",
            b"127.0.0.2:8127",
            b"127.0.0.1:08127",
            b"127.0.0.1:+8127",
            b"127.0.0.1:",
            b"",
            b"\xff:8127",
        ] {
            assert!(!host_allowed(bad, 8127, false), "accepted {:?}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn origins_are_normalized_and_wildcards_refused() {
        assert_eq!(normalize_origin("HTTP://Tauri.Localhost/").expect("ok"), "http://tauri.localhost");
        assert_eq!(normalize_origin("http://127.0.0.1:1420").expect("ok"), "http://127.0.0.1:1420");
        for bad in ["*", "null", "tauri.localhost", "http://a/b", "http://a b", "http://", "://x", "http://user@a"] {
            assert!(normalize_origin(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn preflight_echoes_only_a_plain_header_name_list() {
        let mut h = HeaderMap::new();
        assert_eq!(allowed_request_headers(&h), "authorization, content-type");
        h.insert(header::ACCESS_CONTROL_REQUEST_HEADERS, HeaderValue::from_static("authorization, content-type, x-stainless-os"));
        assert_eq!(allowed_request_headers(&h), "authorization, content-type, x-stainless-os");
        for bad in ["a;b", "x:y", "x\ty"] {
            h.insert(header::ACCESS_CONTROL_REQUEST_HEADERS, HeaderValue::from_str(bad).expect("header"));
            assert_eq!(allowed_request_headers(&h), "authorization, content-type", "{bad:?}");
        }
        h.insert(header::ACCESS_CONTROL_REQUEST_HEADERS, HeaderValue::from_str(&"a".repeat(2000)).expect("header"));
        assert_eq!(allowed_request_headers(&h), "authorization, content-type");
    }

    #[test]
    fn origin_check_against_allowlist() {
        let allowed = vec!["http://tauri.localhost".to_string()];
        let mut h = HeaderMap::new();
        assert_eq!(check_origin(&h, &allowed), OriginCheck::Absent);
        h.insert(header::ORIGIN, HeaderValue::from_static("http://tauri.localhost"));
        assert!(matches!(check_origin(&h, &allowed), OriginCheck::Allowed(_)));
        h.insert(header::ORIGIN, HeaderValue::from_static("https://evil.example"));
        assert_eq!(check_origin(&h, &allowed), OriginCheck::Refused);
        h.insert(header::ORIGIN, HeaderValue::from_static("null"));
        assert_eq!(check_origin(&h, &allowed), OriginCheck::Refused);
        // Nothing is allowed by default.
        h.insert(header::ORIGIN, HeaderValue::from_static("http://tauri.localhost"));
        assert_eq!(check_origin(&h, &[]), OriginCheck::Refused);
    }
}
