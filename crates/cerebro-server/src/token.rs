//! Bearer tokens: the gateway token clients present, and the per-launch key the gateway presents to
//! llama-server.
//!
//! Both are 256 bits from the OS CSPRNG ([`getrandom::fill`]), carried as 64 lowercase hex
//! characters (an HTTP header value, and at least the 32 bytes llama.cpp's `--api-key-file` parser
//! requires per key). The two are always different tokens: the client's token never reaches
//! llama-server, and llama-server's key never leaves Cerebro.
//!
//! The gateway token persists in the DPAPI secret store as its 32 raw bytes
//! ([`cerebro_config::SecretId::ApiBearer`]), so the desktop app and the CLI keep working across
//! restarts. A presented token is compared with [`subtle::ConstantTimeEq`]: the time a mismatch
//! takes does not depend on how many leading characters were right.

use cerebro_config::{Secret, SecretId, SecretStore};
use subtle::ConstantTimeEq as _;
use zeroize::Zeroize as _;

use crate::error::ServerError;

/// Raw token length in bytes (256 bits).
pub const TOKEN_BYTES: usize = 32;

/// Length of a token's hex form, the string sent in `Authorization: Bearer <token>`.
pub const TOKEN_HEX_LEN: usize = TOKEN_BYTES * 2;

/// A 256-bit bearer token held as its hex form, zeroed on drop and never printed.
///
/// `Debug` renders `BearerToken(<redacted>)`; there is no `Display`.
pub struct BearerToken {
    hex: Secret,
}

impl std::fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BearerToken(<redacted>)")
    }
}

impl BearerToken {
    /// A fresh token from the OS CSPRNG.
    ///
    /// # Errors
    /// [`ServerError::Random`] if the OS refuses to produce random bytes.
    pub fn generate() -> Result<Self, ServerError> {
        let mut raw = [0u8; TOKEN_BYTES];
        let filled = getrandom::fill(&mut raw).map_err(|e| ServerError::Random(e.to_string()));
        let token = filled.and_then(|()| Self::from_raw(&raw));
        raw.zeroize();
        token
    }

    /// The token whose raw bytes are `raw`.
    ///
    /// # Errors
    /// [`ServerError::TokenCorrupt`] unless `raw` is exactly [`TOKEN_BYTES`] long.
    pub fn from_raw(raw: &[u8]) -> Result<Self, ServerError> {
        if raw.len() != TOKEN_BYTES {
            return Err(ServerError::TokenCorrupt { found: raw.len(), expected: TOKEN_BYTES });
        }
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut hex = Vec::with_capacity(TOKEN_HEX_LEN);
        for byte in raw {
            hex.push(HEX[usize::from(byte >> 4)]);
            hex.push(HEX[usize::from(byte & 0x0f)]);
        }
        Ok(Self { hex: Secret::new(hex) })
    }

    /// The stored gateway token, creating and storing a fresh one on first use.
    ///
    /// Creation is atomic ([`SecretStore::set_if_absent`]): when two processes start on a fresh
    /// profile together, one stores its token and the other reads that one back, so both serve the
    /// token the store holds.
    ///
    /// # Errors
    /// [`ServerError::Secrets`] if the store cannot be read or written, [`ServerError::TokenCorrupt`]
    /// if what it holds is not a token this crate wrote (use [`BearerToken::rotate`]), or
    /// [`ServerError::Random`].
    pub fn load_or_create(store: &SecretStore) -> Result<Self, ServerError> {
        if let Some(stored) = store.get(SecretId::ApiBearer)? {
            return Self::from_raw(stored.expose_secret());
        }
        let mut raw = [0u8; TOKEN_BYTES];
        let token = getrandom::fill(&mut raw)
            .map_err(|e| ServerError::Random(e.to_string()))
            .and_then(|()| Ok(store.set_if_absent(SecretId::ApiBearer, &Secret::new(raw.to_vec()))?))
            .and_then(|created| {
                if created {
                    return Self::from_raw(&raw);
                }
                // Another process created it first: serve what the store holds.
                match store.get(SecretId::ApiBearer)? {
                    Some(stored) => Self::from_raw(stored.expose_secret()),
                    None => Err(ServerError::TokenCorrupt { found: 0, expected: TOKEN_BYTES }),
                }
            });
        raw.zeroize();
        token
    }

    /// Generate a fresh gateway token and store it, replacing the old one. This changes only the
    /// stored token: a gateway already serving keeps the token it was given until it is told
    /// otherwise, so rotate a running one with [`crate::Serving::rotate_token`] (or hand the result
    /// to [`crate::GatewayHandle::set_token`]), which refuses the old token from the next request on.
    ///
    /// # Errors
    /// [`ServerError::Random`] or [`ServerError::Secrets`].
    pub fn rotate(store: &SecretStore) -> Result<Self, ServerError> {
        let mut raw = [0u8; TOKEN_BYTES];
        getrandom::fill(&mut raw).map_err(|e| ServerError::Random(e.to_string()))?;
        let stored = store.set(SecretId::ApiBearer, &Secret::new(raw.to_vec()));
        let token = stored.map_err(ServerError::from).and_then(|()| Self::from_raw(&raw));
        raw.zeroize();
        token
    }

    /// A second copy of this token (both zeroed on drop).
    pub(crate) fn duplicate(&self) -> Self {
        Self { hex: Secret::new(self.hex.expose_secret().to_vec()) }
    }

    /// The hex form, for the one place it must be shown or sent: the user's client configuration,
    /// or the `Authorization` header to llama-server. Never log it.
    pub fn reveal(&self) -> &str {
        // Built from ASCII hex digits only, so this conversion cannot fail; the empty fallback
        // keeps the function total without a panic path.
        std::str::from_utf8(self.hex.expose_secret()).unwrap_or("")
    }

    /// Whether an `Authorization` header value presents this token as `Bearer <token>`.
    ///
    /// The scheme is matched case-insensitively (RFC 9110); the token itself is compared in
    /// constant time with [`subtle::ConstantTimeEq`]. A wrong length is rejected by `ct_eq` too,
    /// which reveals only the length: always [`TOKEN_HEX_LEN`] for a real token.
    pub fn verify_authorization(&self, header_value: &[u8]) -> bool {
        const SCHEME: &[u8] = b"bearer ";
        if header_value.len() < SCHEME.len()
            || !header_value[..SCHEME.len()].eq_ignore_ascii_case(SCHEME)
        {
            return false;
        }
        let presented = trim_ascii_whitespace(&header_value[SCHEME.len()..]);
        presented.ct_eq(self.hex.expose_secret()).into()
    }
}

/// `presented` without leading or trailing spaces or tabs (header whitespace).
fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while let [b' ' | b'\t', rest @ ..] = bytes {
        bytes = rest;
    }
    while let [rest @ .., b' ' | b'\t'] = bytes {
        bytes = rest;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_of(byte: u8) -> BearerToken {
        BearerToken::from_raw(&[byte; TOKEN_BYTES]).expect("32 bytes")
    }

    #[test]
    fn hex_form_is_64_lowercase_hex_chars() {
        let token = BearerToken::from_raw(&[0xab; TOKEN_BYTES]).expect("32 bytes");
        assert_eq!(token.reveal(), "ab".repeat(TOKEN_BYTES));
        let fresh = BearerToken::generate().expect("OS RNG");
        assert_eq!(fresh.reveal().len(), TOKEN_HEX_LEN);
        assert!(fresh.reveal().bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }

    #[test]
    fn generated_tokens_differ() {
        let a = BearerToken::generate().expect("OS RNG");
        let b = BearerToken::generate().expect("OS RNG");
        assert_ne!(a.reveal(), b.reveal());
    }

    #[test]
    fn wrong_length_raw_token_is_refused() {
        let err = BearerToken::from_raw(&[0; 31]).unwrap_err();
        assert!(matches!(err, ServerError::TokenCorrupt { found: 31, expected: 32 }));
    }

    #[test]
    fn verify_accepts_exact_token_only() {
        let token = token_of(0x11);
        let good = format!("Bearer {}", token.reveal());
        assert!(token.verify_authorization(good.as_bytes()));
        assert!(token.verify_authorization(format!("bearer  {} ", token.reveal()).as_bytes()));
        assert!(token.verify_authorization(format!("BEARER {}", token.reveal()).as_bytes()));

        // Wrong in the first character, wrong in the last, one short, one long, wrong scheme.
        let mut first = token.reveal().to_string();
        first.replace_range(0..1, "2");
        let mut last = token.reveal().to_string();
        last.replace_range(TOKEN_HEX_LEN - 1.., "2");
        for bad in [
            format!("Bearer {first}"),
            format!("Bearer {last}"),
            format!("Bearer {}", &token.reveal()[1..]),
            format!("Bearer {}0", token.reveal()),
            format!("Basic {}", token.reveal()),
            token.reveal().to_string(),
            "Bearer ".to_string(),
            String::new(),
        ] {
            assert!(!token.verify_authorization(bad.as_bytes()), "accepted {bad:?}");
        }
        // Another token's hex is refused.
        assert!(!token.verify_authorization(format!("Bearer {}", token_of(0x12).reveal()).as_bytes()));
    }

    /// The comparison of the presented token must go through `subtle`'s constant-time equality,
    /// never `==` or a short-circuiting slice compare. This reads the function's own source, so
    /// replacing `ct_eq` with `==` (which every functional test above would still pass) fails here.
    #[test]
    fn presented_token_is_compared_in_constant_time() {
        let source = include_str!("token.rs");
        let start = source.find("pub fn verify_authorization").expect("function present");
        let end = start + source[start..].find("\n    }").expect("function end");
        let body = &source[start..end];
        assert!(body.contains("presented.ct_eq(self.hex.expose_secret())"), "{body}");
        assert!(!body.contains("== self.hex") && !body.contains("self.hex.expose_secret() =="), "{body}");
        assert!(source.contains("use subtle::ConstantTimeEq as _;"));
    }

    #[test]
    fn debug_never_prints_the_token() {
        let token = token_of(0x5a);
        let printed = format!("{token:?}");
        assert_eq!(printed, "BearerToken(<redacted>)");
        assert!(!printed.contains(&token.reveal()[..8]));
    }

    #[cfg(windows)]
    #[test]
    fn store_round_trip_creates_once_then_reloads_and_rotates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let first = BearerToken::load_or_create(&store).expect("create");
        let again = BearerToken::load_or_create(&store).expect("reload");
        assert_eq!(first.reveal(), again.reveal());
        let rotated = BearerToken::rotate(&store).expect("rotate");
        assert_ne!(rotated.reveal(), first.reveal());
        let reloaded = BearerToken::load_or_create(&store).expect("reload rotated");
        assert_eq!(reloaded.reveal(), rotated.reveal());

        // Two processes starting together on a fresh profile serve the same, stored token.
        let fresh = tempfile::tempdir().expect("tempdir");
        drop(SecretStore::open(fresh.path()).expect("open store"));
        let racers: Vec<_> = (0..6)
            .map(|_| {
                let path = fresh.path().to_path_buf();
                std::thread::spawn(move || {
                    let store = SecretStore::open(&path).expect("open store");
                    BearerToken::load_or_create(&store).expect("create").reveal().to_string()
                })
            })
            .collect();
        let tokens: Vec<String> = racers.into_iter().map(|h| h.join().expect("thread")).collect();
        let stored = BearerToken::load_or_create(&SecretStore::open(fresh.path()).expect("open")).expect("reload");
        assert!(tokens.iter().all(|t| t == stored.reveal()), "every racer serves the stored token");

        // A blob of the wrong size is reported, never silently replaced.
        store.set(SecretId::ApiBearer, &Secret::new(vec![1; 16])).expect("plant short token");
        let err = BearerToken::load_or_create(&store).unwrap_err();
        assert!(matches!(err, ServerError::TokenCorrupt { found: 16, .. }), "{err}");
    }
}
