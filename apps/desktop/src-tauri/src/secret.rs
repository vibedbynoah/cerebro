//! 256-bit secrets, for the loopback API's bearer token.
//!
//! Only vetted primitives: the bytes come from the operating system's CSPRNG (`getrandom`), and a
//! presented token is compared in constant time (`subtle`), so response timing reveals nothing
//! about how many leading bytes were right. The bytes are wiped (`zeroize`, which the optimiser
//! cannot elide) when the secret is dropped. Nothing here is hand-rolled cryptography.

use std::fmt;

use subtle::ConstantTimeEq;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Length of every secret, in bytes (256 bits).
pub const SECRET_LEN: usize = 32;

/// A random 256-bit secret. `Debug` never prints it, and dropping it wipes the bytes. Deliberately
/// not `Clone` or `Copy`, so no unwiped copy of the bytes can be made through the type.
pub struct Secret([u8; SECRET_LEN]);

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// Marker: `Secret` wipes itself on drop (the `Drop` impl above).
impl ZeroizeOnDrop for Secret {}

impl Secret {
    /// Draws a fresh secret from the OS CSPRNG.
    ///
    /// # Errors
    /// Returns the OS error if the random source is unavailable; callers must not fall back to a
    /// weaker source.
    pub fn generate() -> Result<Self, getrandom::Error> {
        let mut bytes = [0u8; SECRET_LEN];
        if let Err(error) = getrandom::fill(&mut bytes) {
            bytes.zeroize();
            return Err(error);
        }
        let secret = Self(bytes);
        // `[u8; N]` is `Copy`: the local still holds the bytes, so wipe it too.
        bytes.zeroize();
        Ok(secret)
    }

    /// Lower-case hex, for handing the token to a client once (64 characters).
    pub fn to_hex(&self) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(SECRET_LEN * 2);
        for &b in &self.0 {
            out.push(char::from(DIGITS[usize::from(b >> 4)]));
            out.push(char::from(DIGITS[usize::from(b & 0x0f)]));
        }
        out
    }

    /// Constant-time check of a presented token. Any length other than [`SECRET_LEN`] is rejected
    /// (the length is public).
    pub fn matches(&self, presented: &[u8]) -> bool {
        presented.len() == SECRET_LEN && bool::from(self.0.as_slice().ct_eq(presented))
    }

    /// Constant-time check of a hex token as a client sends it. Malformed hex never matches.
    pub fn matches_hex(&self, presented: &str) -> bool {
        // The decoded copy is what a client sent (the right token, if it matches): wipe it too.
        decode_hex(presented).is_some_and(|mut bytes| {
            let ok = self.matches(&bytes);
            bytes.zeroize();
            ok
        })
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn decode_hex(s: &str) -> Option<[u8; SECRET_LEN]> {
    let raw = s.as_bytes();
    if raw.len() != SECRET_LEN * 2 {
        return None;
    }
    let mut out = [0u8; SECRET_LEN];
    for (slot, pair) in out.iter_mut().zip(raw.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Secret {
        Secret::generate().expect("OS CSPRNG available in tests")
    }

    #[test]
    fn secrets_are_256_bit_and_distinct() {
        let (a, b) = (fresh(), fresh());
        assert_eq!(a.to_hex().len(), 64);
        assert_ne!(a.to_hex(), b.to_hex());
    }

    #[test]
    fn only_the_exact_token_matches() {
        let s = fresh();
        let hex = s.to_hex();
        assert!(s.matches_hex(&hex));
        assert!(s.matches_hex(&hex.to_uppercase()));
        let last = if hex.ends_with('0') { '1' } else { '0' };
        assert!(!s.matches_hex(&format!("{}{last}", &hex[..63])));
        assert!(!s.matches_hex(&hex[..62]));
        assert!(!s.matches_hex(&format!("{hex}00")));
        assert!(!s.matches_hex(&"zz".repeat(32)));
        assert!(!s.matches(&[]));
    }

    /// Compile-time check: this only builds while `Secret` is `ZeroizeOnDrop`.
    #[test]
    fn secret_wipes_itself_on_drop() {
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}
        assert_zeroize_on_drop::<Secret>();
    }

    #[test]
    fn debug_never_prints_the_secret() {
        let s = fresh();
        assert_eq!(format!("{s:?}"), "Secret(<redacted>)");
    }
}
