//! Ed25519 signature check for update packages.
//!
//! An update is applied only if its bytes verify against the release public key that is compiled
//! into the app. Verification uses `ed25519-dalek`'s `verify_strict`, which also rejects weak
//! (small-order) keys and non-canonical signatures, so one payload cannot carry two valid
//! signatures. Nothing here is hand-rolled; this module only fixes the sizes and the error shape.
//!
//! The updater itself is not wired yet (the app makes no network requests in this build); the check
//! lives here so the first updater lands on a tested primitive instead of writing its own.

use ed25519_dalek::{Signature, VerifyingKey};

/// Length of an Ed25519 public key, in bytes.
pub const PUBLIC_KEY_LEN: usize = 32;
/// Length of an Ed25519 signature, in bytes.
pub const SIGNATURE_LEN: usize = 64;

/// Why an update was refused. Deliberately coarse: the reason is logged, never acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateSigError {
    /// The compiled-in public key is not a valid, strong Ed25519 point.
    BadKey,
    /// The signature is not exactly [`SIGNATURE_LEN`] bytes.
    BadSignatureLength,
    /// The signature does not verify for these bytes under this key.
    Mismatch,
}

impl std::fmt::Display for UpdateSigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BadKey => "the update public key is invalid",
            Self::BadSignatureLength => "the update signature has the wrong length",
            Self::Mismatch => "the update signature does not match the package",
        })
    }
}

impl std::error::Error for UpdateSigError {}

/// Verifies `signature` over the whole `payload` with `public_key`.
///
/// # Errors
/// [`UpdateSigError`] for an invalid key, a malformed signature, or any mismatch.
pub fn verify_update(public_key: &[u8; PUBLIC_KEY_LEN], payload: &[u8], signature: &[u8]) -> Result<(), UpdateSigError> {
    let key = VerifyingKey::from_bytes(public_key).map_err(|_| UpdateSigError::BadKey)?;
    if key.is_weak() {
        return Err(UpdateSigError::BadKey);
    }
    let bytes: [u8; SIGNATURE_LEN] = signature.try_into().map_err(|_| UpdateSigError::BadSignatureLength)?;
    key.verify_strict(payload, &Signature::from_bytes(&bytes)).map_err(|_| UpdateSigError::Mismatch)
}

#[cfg(test)]
mod tests {
    use super::{verify_update, UpdateSigError};
    use ed25519_dalek::{Signer, SigningKey};

    fn keypair() -> (SigningKey, [u8; 32]) {
        // A fixed test key: tests must be deterministic, and this key signs nothing real.
        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let public = signing.verifying_key().to_bytes();
        (signing, public)
    }

    #[test]
    fn a_genuine_signature_verifies() {
        let (signing, public) = keypair();
        let payload = b"cerebro-0.2.0-x64-setup.exe contents";
        let sig = signing.sign(payload).to_bytes();
        assert_eq!(verify_update(&public, payload, &sig), Ok(()));
    }

    #[test]
    fn any_changed_byte_is_refused() {
        let (signing, public) = keypair();
        let payload = b"package bytes".to_vec();
        let sig = signing.sign(&payload).to_bytes();

        let mut tampered = payload.clone();
        tampered[0] ^= 1;
        assert_eq!(verify_update(&public, &tampered, &sig), Err(UpdateSigError::Mismatch));

        let mut bad_sig = sig;
        bad_sig[10] ^= 1;
        assert_eq!(verify_update(&public, &payload, &bad_sig), Err(UpdateSigError::Mismatch));
    }

    #[test]
    fn another_key_is_refused() {
        let (signing, _) = keypair();
        let other = SigningKey::from_bytes(&[9u8; 32]).verifying_key().to_bytes();
        let sig = signing.sign(b"x").to_bytes();
        assert_eq!(verify_update(&other, b"x", &sig), Err(UpdateSigError::Mismatch));
    }

    #[test]
    fn malformed_inputs_are_refused() {
        let (signing, public) = keypair();
        let sig = signing.sign(b"x").to_bytes();
        assert_eq!(verify_update(&public, b"x", &sig[..63]), Err(UpdateSigError::BadSignatureLength));
        assert_eq!(verify_update(&public, b"x", &[]), Err(UpdateSigError::BadSignatureLength));
        // The identity point is a small-order (weak) key.
        let mut identity = [0u8; 32];
        identity[0] = 1;
        assert_eq!(verify_update(&identity, b"x", &sig), Err(UpdateSigError::BadKey));
    }
}
