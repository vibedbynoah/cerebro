//! Pure, I/O-free codec for one stored line.
//!
//! Kept separate from [`crate::store`] so the hash-chain and parsing rules are unit-testable
//! directly on byte slices, without a filesystem: [`encode_line`] turns a [`Record`] into the
//! exact bytes [`crate::store::Store::append`] writes (minus the trailing newline, which the
//! caller adds, since it may also need to prepend a repair newline), and [`decode_line`] parses
//! and fully verifies one line's bytes back into a [`Record`].
//!
//! # On-disk line shape
//!
//! ```json
//! {"prev_hash":null,"record":{...},"record_hash":"<sha256 hex>"}
//! ```
//!
//! `record_hash` covers the record itself (`sha256(prev_hash_or_empty || 0x00 || raw record
//! bytes)`), not just the link to the record before it -- see the [`crate::store`] module
//! documentation for what that does and does not let this crate detect.

use serde::Deserialize;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

use crate::error::StoreError;
use crate::record::{cap_echo, Record, RecordWire, SCHEMA_VERSION};

/// One line's outer envelope, borrowed from the line buffer.
///
/// `record` is a [`RawValue`] specifically so hashing and the nested parse both operate on the
/// exact bytes that were written, never a re-serialization of them: re-serializing could reorder
/// map keys or reformat floats and silently break the chain or the record's own self-hash.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredLineWire<'a> {
    prev_hash: Option<String>,
    #[serde(borrow)]
    record: &'a RawValue,
    record_hash: String,
}

/// The link hash for one line: `sha256(prev_hash_or_empty || 0x00 || raw_record_bytes)`.
///
/// The `0x00` domain separator between `prev_hash` and the record bytes means a `prev_hash` of
/// `"ab"` followed by record bytes `"c"` cannot collide with `prev_hash` `"a"` followed by record
/// bytes `"bc"` -- plain concatenation would let that boundary shift.
pub(crate) fn link_hash(prev_hash: Option<&str>, raw_record: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.unwrap_or("").as_bytes());
    hasher.update([0u8]);
    hasher.update(raw_record);
    format!("{:x}", hasher.finalize())
}

/// Encode one record as the bytes that belong on its own line, **without** a trailing newline.
///
/// Returns the encoded bytes and this line's own link hash, which is the `prev_hash` the next
/// appended line must carry.
///
/// # Errors
/// [`StoreError::Unserializable`] if `record` cannot be serialized to JSON.
pub(crate) fn encode_line(prev_hash: Option<&str>, record: &Record) -> Result<(Vec<u8>, String), StoreError> {
    let raw_record = serde_json::to_vec(record).map_err(|e| StoreError::Unserializable { detail: e.to_string() })?;
    let hash = link_hash(prev_hash, &raw_record);

    let mut out = Vec::with_capacity(raw_record.len() + hash.len() + 32);
    out.extend_from_slice(b"{\"prev_hash\":");
    match prev_hash {
        // Every prev_hash this crate ever writes is its own lowercase-hex `{:x}` output, so it
        // contains no character that needs JSON string escaping.
        Some(h) => {
            out.push(b'"');
            out.extend_from_slice(h.as_bytes());
            out.push(b'"');
        }
        None => out.extend_from_slice(b"null"),
    }
    out.extend_from_slice(b",\"record\":");
    out.extend_from_slice(&raw_record);
    out.extend_from_slice(b",\"record_hash\":\"");
    out.extend_from_slice(hash.as_bytes());
    out.extend_from_slice(b"\"}");
    Ok((out, hash))
}

/// A decoded, fully verified line.
#[derive(Debug)]
pub(crate) struct DecodedLine {
    /// The validated record.
    pub record: Record,
    /// This line's own link hash, to chain the next line onto.
    pub record_hash: String,
}

/// Parse and fully verify one line's content (no trailing newline), against the hash chain built
/// so far.
///
/// # Errors
/// - [`StoreError::MalformedLine`] if `content` is not the expected JSON envelope, or the nested
///   `record` does not match [`RecordWire`]'s shape (missing/extra/mistyped field).
/// - [`StoreError::ChainBroken`] if `prev_hash` does not match `expected_prev`, or `record_hash`
///   does not match the hash recomputed from this line's own bytes -- this crate does not
///   distinguish "does not chain onto the record before it" from "this record's own bytes were
///   altered"; both are a broken link.
/// - [`StoreError::UnsupportedSchemaVersion`] if the nested record's `schema_version` is not one
///   this build understands.
/// - [`StoreError::InvalidRecord`] if the nested record is well-formed JSON but fails
///   [`crate::error::RecordError`] validation.
pub(crate) fn decode_line(content: &[u8], index: usize, expected_prev: Option<&str>) -> Result<DecodedLine, StoreError> {
    let wire: StoredLineWire = serde_json::from_slice(content).map_err(|e| StoreError::MalformedLine { index, detail: e.to_string() })?;

    let raw_record_bytes = wire.record.get().as_bytes();
    let record_hash = link_hash(wire.prev_hash.as_deref(), raw_record_bytes);

    if wire.prev_hash.as_deref() != expected_prev {
        // `expected_prev` is this crate's own previously-computed hash; only `found` came off
        // disk, so only `found` needs the untrusted-echo cap.
        return Err(StoreError::ChainBroken {
            index,
            expected: expected_prev.map(str::to_string),
            found: wire.prev_hash.as_deref().map(cap_echo),
        });
    }
    if wire.record_hash != record_hash {
        // `expected` here is `record_hash`, this crate's own recomputation; `found` is the
        // attacker-controlled string read from the line.
        return Err(StoreError::ChainBroken { index, expected: Some(record_hash), found: Some(cap_echo(&wire.record_hash)) });
    }

    // Check the schema version from a permissive peek *before* the strict, `deny_unknown_fields`
    // parse below: an older or newer schema's shape may not satisfy the current `RecordWire` at
    // all (a required field can be added or removed between versions), and that structural
    // mismatch must be reported as `UnsupportedSchemaVersion`, not misreported as a generic
    // `MalformedLine` just because this build's strict shape does not happen to match it.
    #[derive(Deserialize)]
    struct SchemaVersionPeek {
        schema_version: u32,
    }
    let peeked: SchemaVersionPeek =
        serde_json::from_str(wire.record.get()).map_err(|e| StoreError::MalformedLine { index, detail: e.to_string() })?;
    if peeked.schema_version != SCHEMA_VERSION {
        return Err(StoreError::UnsupportedSchemaVersion { index, found: peeked.schema_version, supported: SCHEMA_VERSION });
    }

    let record_wire: RecordWire =
        serde_json::from_str(wire.record.get()).map_err(|e| StoreError::MalformedLine { index, detail: e.to_string() })?;
    // `RecordWire::schema_version` exists only so `deny_unknown_fields` still accepts the JSON's
    // `schema_version` key; the version was already checked above via the permissive peek, and by
    // construction (same bytes, same field) it cannot disagree here.
    debug_assert_eq!(record_wire.schema_version, SCHEMA_VERSION);
    let record = Record::from_wire(record_wire).map_err(|source| StoreError::InvalidRecord { index, source })?;
    Ok(DecodedLine { record, record_hash })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MeasurementSummary, MeasurementTriple, ProvenanceKey, SamplerProfile};

    fn summary(median: f64) -> MeasurementSummary {
        MeasurementSummary::new(3, median, median - 1.0, median + 1.0, 1.0, median - 0.5, median + 0.5).unwrap()
    }

    fn record() -> Record {
        let provenance =
            ProvenanceKey::new("c".repeat(64), "a".repeat(64), "b".repeat(64), Some(10751), None, None, None, Some(7113)).unwrap();
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(2.5), 128).unwrap();
        Record::new(provenance, sampler, vec!["llama-server".into()], triple, None, None, 1_726_500_000_000).unwrap()
    }

    #[test]
    fn encode_then_decode_round_trips() {
        let r = record();
        let (bytes, hash) = encode_line(None, &r).unwrap();
        let decoded = decode_line(&bytes, 0, None).unwrap();
        assert_eq!(decoded.record, r);
        assert_eq!(decoded.record_hash, hash);
    }

    #[test]
    fn chained_line_requires_the_right_prev_hash() {
        let r = record();
        let (bytes, hash) = encode_line(Some("d".repeat(64).as_str()), &r).unwrap();
        assert!(decode_line(&bytes, 1, Some(&"d".repeat(64))).is_ok());
        let err = decode_line(&bytes, 1, Some(&"e".repeat(64))).unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 1, .. }), "got {err:?}");
        let _ = hash;
    }

    #[test]
    fn editing_the_record_bytes_breaks_its_own_self_hash() {
        let r = record();
        let (mut bytes, _hash) = encode_line(None, &r).unwrap();
        // Flip a digit inside the embedded record JSON without touching record_hash: the
        // self-hash must catch this even though there is no "next line" to carry a mismatched
        // prev_hash -- this is the fix for editing the *last* record in a file.
        let needle = b"1726500000000";
        let pos = bytes.windows(needle.len()).position(|w| w == needle).expect("timestamp must appear literally");
        bytes[pos] = b'9';
        let err = decode_line(&bytes, 0, None).unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 0, .. }), "got {err:?}");
    }

    #[test]
    fn unknown_top_level_field_is_malformed() {
        let bad = br#"{"prev_hash":null,"record":{},"record_hash":"x","evil":true}"#;
        let err = decode_line(bad, 0, None).unwrap_err();
        assert!(matches!(err, StoreError::MalformedLine { index: 0, .. }), "got {err:?}");
    }

    #[test]
    fn non_object_top_level_line_is_malformed() {
        // The stored-line envelope itself must be a JSON object with exactly
        // `prev_hash`/`record`/`record_hash`; none of these can match that shape.
        for bad_line in [br#"[1,2]"#.as_slice(), br#"42"#.as_slice(), br#""a string""#.as_slice(), br#"null"#.as_slice()] {
            let err = decode_line(bad_line, 0, None).unwrap_err();
            assert!(matches!(err, StoreError::MalformedLine { index: 0, .. }), "line {bad_line:?} got {err:?}");
        }
    }

    #[test]
    fn chain_error_caps_the_echoed_hash_from_a_hostile_line() {
        // A results file is hostile input: a 100,000-character `prev_hash` (still comfortably
        // under `MAX_RECORD_BYTES`) must not turn into a 100,000-character error message.
        let mut line = Vec::new();
        line.extend_from_slice(br#"{"prev_hash":""#);
        line.extend_from_slice("a".repeat(100_000).as_bytes());
        line.extend_from_slice(br#"","record":{},"record_hash":"x"}"#);
        match decode_line(&line, 0, None).unwrap_err() {
            StoreError::ChainBroken { found: Some(f), .. } => assert_eq!(f.len(), 80, "echoed hash must be capped"),
            other => panic!("expected ChainBroken, got {other:?}"),
        }
    }

    #[test]
    fn chain_error_caps_the_echoed_record_hash_from_a_hostile_line() {
        // The second untrusted echo: a genuinely chained line (correct prev_hash) whose own
        // `record_hash` claim is oversized must also be capped.
        let mut line = Vec::new();
        line.extend_from_slice(br#"{"prev_hash":null,"record":{},"record_hash":""#);
        line.extend_from_slice("b".repeat(100_000).as_bytes());
        line.extend_from_slice(br#""}"#);
        match decode_line(&line, 0, None).unwrap_err() {
            StoreError::ChainBroken { found: Some(f), .. } => assert_eq!(f.len(), 80, "echoed hash must be capped"),
            other => panic!("expected ChainBroken, got {other:?}"),
        }
    }
}
