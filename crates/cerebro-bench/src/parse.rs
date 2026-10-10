//! Deserializing llama-bench's `-o json` output.
//!
//! `docs/ARCHITECTURE.md` section 6 designates this the primary measurement source: parse
//! `samples_ns`/`samples_ts` — the raw per-repetition samples — never `avg_ns`/`avg_ts`, because
//! only the raw samples permit the nonparametric statistics in [`crate::stats`]. Everything else
//! llama-bench echoes about how it was invoked is kept too, because it is the provenance key that
//! makes one measurement comparable to another (section 6's closing point: a self-built fork of
//! the same commit measured 3-5x slower than the official release on this machine).
//!
//! Deserialization is deliberately permissive: fields llama-bench might omit in an older or newer
//! build are `Option`, and fields this crate does not know about at all are silently ignored by
//! `serde` (no `deny_unknown_fields` anywhere in this module) rather than rejected. What *is*
//! rejected, in [`parse_records`], is a document that is structurally impossible to summarize —
//! mismatched or empty sample arrays — because those failures belong at the parse boundary rather
//! than surfacing two layers downstream as a confusing statistics error.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::ParseError;

/// Most records a single `-o json` document may carry.
///
/// A tampered or corrupted results file could otherwise carry an unbounded number of records;
/// each one becomes a bootstrap computation in [`crate::stats::assemble_triples`], so this bounds
/// one of the two factors that multiply into that cost. The other, repetitions per record, is
/// bounded by [`crate::stats::MAX_REPETITIONS`].
pub const MAX_RECORDS: usize = 100_000;

/// One row of llama-bench's `-o json` array: one `(configuration, test)` measurement.
///
/// Field names match the JSON keys llama-bench b10751 emits exactly, verified by running
/// `llama-bench.exe -o json` against the local CPU fixture (see `tests/fixtures/`). One rename is
/// necessary: llama-bench's own field is `model_n_params`, kept here as `n_params` to match this
/// crate's naming for the corresponding `-p`/`-n`/`-d` triple of `n_prompt`/`n_gen`/`n_depth`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BenchRecord {
    // ---- backend provenance -------------------------------------------------------------
    /// Short commit hash of the llama.cpp build that produced this record.
    pub build_commit: String,
    /// Upstream build number of the llama.cpp build that produced this record.
    pub build_number: u32,
    /// Comma-separated backend names the binary was compiled with (e.g. `"CUDA"`).
    #[serde(default)]
    pub backends: Option<String>,

    // ---- model identity -------------------------------------------------------------------
    /// Path to the model file, exactly as passed to `-m`.
    pub model_filename: String,
    /// Human-readable model type string llama-bench prints (e.g. `"qwen2 1B F16"`).
    #[serde(default)]
    pub model_type: Option<String>,
    /// Model file size in bytes.
    #[serde(default)]
    pub model_size: Option<u64>,
    /// Model parameter count. Renamed from llama-bench's `model_n_params`.
    #[serde(rename = "model_n_params", default)]
    pub n_params: Option<u64>,

    // ---- offload / cache configuration ------------------------------------------------------
    /// `-ngl` value in effect. `-1` is llama-bench's "let the fitter decide" sentinel.
    #[serde(default)]
    pub n_gpu_layers: Option<i64>,
    /// `-ncmoe` value in effect. `Option` because older llama-bench builds predate this flag.
    #[serde(default)]
    pub n_cpu_moe: Option<i64>,
    /// `-ctk` value in effect (e.g. `"f16"`, `"q8_0"`).
    #[serde(rename = "type_k", default)]
    pub type_k: Option<String>,
    /// `-ctv` value in effect.
    #[serde(rename = "type_v", default)]
    pub type_v: Option<String>,
    /// `-fa` value in effect, as llama-bench encodes it: `-1` auto, `0` off, `1` on.
    #[serde(default)]
    pub flash_attn: Option<i64>,

    // ---- CPU / batch configuration -----------------------------------------------------------
    /// `-t` value in effect.
    #[serde(default)]
    pub n_threads: Option<u32>,
    /// `-C`/`--cpu-mask` value in effect, as llama-bench renders it (e.g. `"0x0"`).
    #[serde(default)]
    pub cpu_mask: Option<String>,
    /// `-b` value in effect.
    #[serde(default)]
    pub n_batch: Option<u32>,
    /// `-ub` value in effect.
    #[serde(default)]
    pub n_ubatch: Option<u32>,
    /// `-dev` value in effect, as llama-bench renders it (e.g. `"none"`, a device list).
    #[serde(default)]
    pub devices: Option<String>,

    // ---- test dimensions: required, drive classification and grouping ------------------------
    /// Prompt token count for this test (`-p`). Nonzero with `n_gen == 0` marks a prefill test.
    pub n_prompt: u64,
    /// Generation token count for this test (`-n`). Nonzero marks a decode test.
    pub n_gen: u64,
    /// Context depth this test ran at (`-d`). Defaults to `0` to match llama-bench's own default,
    /// covering builds where a depth-0 test omits the field entirely rather than emitting `0`.
    #[serde(default)]
    pub n_depth: u64,

    // ---- raw per-repetition samples: the only numbers `stats` trusts -------------------------
    /// Wall-clock duration of each repetition, in nanoseconds, in run order.
    pub samples_ns: Vec<u64>,
    /// Throughput of each repetition, in tokens/second, in run order and index-aligned with
    /// `samples_ns`.
    pub samples_ts: Vec<f64>,

    // ---- everything else llama-bench echoes ---------------------------------------------------
    /// Every JSON field on this record that the struct fields above do not already capture, kept
    /// verbatim as `serde_json::Value`.
    ///
    /// `stats::ConfigKey` folds this into the configuration identity (minus a short list of
    /// result-only fields such as `test_time`) precisely so that a field this crate has never
    /// heard of — the b10751 fixture alone echoes `tensor_buft_overrides`, `split_mode`,
    /// `main_gpu`, `no_kv_offload`, `tensor_split`, `load_mode`, `embeddings`, `no_op_offload`,
    /// `no_host`, `poll`, and `cpu_strict` — can only ever *split* two configurations that differ
    /// in it apart, never silently merge them under one key.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Parse a llama-bench `-o json` document into its records.
///
/// This is the only fallible boundary in this module. Beyond syntactic JSON validity, two
/// structural invariants are checked because a value that violates them cannot be summarized
/// safely by [`crate::stats`]: `samples_ns` and `samples_ts` must have equal, nonzero length in
/// every record.
pub fn parse_records(json: &str) -> Result<Vec<BenchRecord>, ParseError> {
    let records: Vec<BenchRecord> =
        serde_json::from_str(json).map_err(|e| ParseError::Malformed { detail: e.to_string() })?;

    if records.len() > MAX_RECORDS {
        return Err(ParseError::TooManyRecords { len: records.len(), max: MAX_RECORDS });
    }

    for (index, record) in records.iter().enumerate() {
        // Checked before the length-mismatch check below: a record hand-edited or duplicated into
        // having an absurd repetition count is corrupt regardless of whether its two sample
        // arrays happen to still agree in length.
        if record.samples_ts.len() > crate::stats::MAX_REPETITIONS {
            return Err(ParseError::TooManyRepetitions {
                index,
                len: record.samples_ts.len(),
                max: crate::stats::MAX_REPETITIONS,
            });
        }
        if record.samples_ns.len() != record.samples_ts.len() {
            return Err(ParseError::SampleLengthMismatch {
                index,
                ns_len: record.samples_ns.len(),
                ts_len: record.samples_ts.len(),
            });
        }
        if record.samples_ns.is_empty() {
            return Err(ParseError::EmptySamples { index });
        }
        // Zero, negative, or non-finite throughput is structurally impossible: llama-bench divides
        // a token count by a positive elapsed time, so a value like this means the document was
        // hand-edited or corrupted. `f64::is_finite` also excludes NaN, which `serde_json` cannot
        // itself produce from standard JSON but which a record built directly in code could carry.
        if let Some((sample, value)) =
            record.samples_ts.iter().enumerate().find(|(_, v)| !(v.is_finite() && **v > 0.0))
        {
            return Err(ParseError::InvalidSample { index, sample, value: value.to_string() });
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_real_cpu_fixture() {
        // Generated by running llama-bench.exe on the tiny fixture model, CPU-only:
        // llama-bench.exe -o json -r 4 -p 64 -n 32 -d 0,128 -dev none
        //
        // Sanitized after generation: `model_filename` was rewritten from the developer's real
        // absolute home-directory path to `models/bastion-control-0.5b-f16.gguf` in all 4
        // records, since no test needs the real path. `cpu_info`/`gpu_info` (real hardware model
        // strings) and every sample/configuration value are left exactly as llama-bench emitted
        // them.
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).expect("real fixture must parse");
        assert_eq!(records.len(), 4, "one record per (test, depth) combination");

        let prefill_d0 = &records[0];
        assert_eq!(prefill_d0.n_prompt, 64);
        assert_eq!(prefill_d0.n_gen, 0);
        assert_eq!(prefill_d0.n_depth, 0);
        assert_eq!(prefill_d0.samples_ns.len(), 4);
        assert_eq!(prefill_d0.samples_ts.len(), 4);
        assert_eq!(prefill_d0.build_commit, "3466812d1");
        assert_eq!(prefill_d0.build_number, 10751);
    }

    #[test]
    fn renamed_field_reads_model_n_params() {
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).expect("real fixture must parse");
        assert_eq!(records[0].n_params, Some(494_032_768));
    }

    #[test]
    fn tolerates_a_field_it_does_not_know_about() {
        // `cpu_info`/`gpu_info`/`test_time`/`avg_ns`/... appear in real output and are not modeled
        // here at all; their presence must not be a parse error.
        let json = r#"[{
            "build_commit": "deadbeef",
            "build_number": 1,
            "model_filename": "m.gguf",
            "n_prompt": 64,
            "n_gen": 0,
            "n_depth": 0,
            "samples_ns": [1, 2, 3, 4],
            "samples_ts": [1.0, 2.0, 3.0, 4.0],
            "a_field_from_a_future_llama_bench_build": {"nested": true}
        }]"#;
        let records = parse_records(json).expect("unknown fields must be tolerated");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].extra.get("a_field_from_a_future_llama_bench_build"),
            Some(&serde_json::json!({"nested": true}))
        );
    }

    #[test]
    fn flatten_does_not_duplicate_a_field_the_struct_already_models() {
        // `model_n_params` is consumed by the renamed `n_params` field; a naive combination of a
        // renamed field with `#[serde(flatten)]` could leave it captured twice.
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).expect("real fixture must parse");
        assert!(
            !records[0].extra.contains_key("model_n_params"),
            "model_n_params must be consumed by the renamed n_params field, not left in extra too"
        );
        assert!(!records[0].extra.contains_key("n_prompt"), "n_prompt is a named field, not extra");
        assert!(
            records[0].extra.contains_key("tensor_buft_overrides"),
            "a genuinely unmodeled field must still land in extra"
        );
    }

    #[test]
    fn rejects_non_positive_throughput_sample() {
        let json = r#"[{
            "build_commit": "deadbeef",
            "build_number": 1,
            "model_filename": "m.gguf",
            "n_prompt": 64,
            "n_gen": 0,
            "n_depth": 0,
            "samples_ns": [1, 2, 3, 4],
            "samples_ts": [1.0, 2.0, -1.0, 4.0]
        }]"#;
        assert_eq!(
            parse_records(json),
            Err(ParseError::InvalidSample { index: 0, sample: 2, value: "-1".to_string() })
        );
    }

    #[test]
    fn tolerates_a_missing_optional_config_field() {
        // n_cpu_moe predates some builds; every other optional field must default the same way.
        let json = r#"[{
            "build_commit": "deadbeef",
            "build_number": 1,
            "model_filename": "m.gguf",
            "n_prompt": 0,
            "n_gen": 32,
            "samples_ns": [1, 2, 3, 4],
            "samples_ts": [1.0, 2.0, 3.0, 4.0]
        }]"#;
        let records = parse_records(json).expect("optional fields must default");
        assert_eq!(records[0].n_cpu_moe, None);
        assert_eq!(records[0].n_depth, 0, "missing n_depth defaults to llama-bench's own default");
    }

    #[test]
    fn rejects_top_level_object_instead_of_array() {
        let json = include_str!("../tests/fixtures/malformed/not_an_array.json");
        assert!(matches!(parse_records(json), Err(ParseError::Malformed { .. })));
    }

    #[test]
    fn rejects_truncated_json() {
        let json = include_str!("../tests/fixtures/malformed/truncated.json");
        assert!(matches!(parse_records(json), Err(ParseError::Malformed { .. })));
    }

    #[test]
    fn rejects_wrong_type_for_a_required_field() {
        let json = include_str!("../tests/fixtures/malformed/wrong_type_n_prompt.json");
        assert!(matches!(parse_records(json), Err(ParseError::Malformed { .. })));
    }

    #[test]
    fn rejects_missing_required_field() {
        let json = include_str!("../tests/fixtures/malformed/missing_samples_ts.json");
        assert!(matches!(parse_records(json), Err(ParseError::Malformed { .. })));
    }

    #[test]
    fn rejects_mismatched_sample_lengths() {
        let json = include_str!("../tests/fixtures/malformed/mismatched_sample_lengths.json");
        assert_eq!(
            parse_records(json),
            Err(ParseError::SampleLengthMismatch { index: 0, ns_len: 4, ts_len: 3 })
        );
    }

    #[test]
    fn rejects_empty_samples() {
        let json = include_str!("../tests/fixtures/malformed/empty_samples.json");
        assert_eq!(parse_records(json), Err(ParseError::EmptySamples { index: 0 }));
    }

    #[test]
    fn parse_rejects_record_over_repetition_cap() {
        // One record with MAX_REPETITIONS + 1 samples: a bootstrap over this many kept samples,
        // multiplied by the iteration count, is exactly the hang MAX_REPETITIONS exists to bound.
        let too_many = crate::stats::MAX_REPETITIONS + 1;
        let ns: Vec<String> = (0..too_many).map(|i| i.to_string()).collect();
        let ts: Vec<String> = (0..too_many).map(|_| "1.0".to_string()).collect();
        let json = format!(
            r#"[{{
                "build_commit": "deadbeef",
                "build_number": 1,
                "model_filename": "m.gguf",
                "n_prompt": 64,
                "n_gen": 0,
                "n_depth": 0,
                "samples_ns": [{}],
                "samples_ts": [{}]
            }}]"#,
            ns.join(","),
            ts.join(",")
        );
        assert_eq!(
            parse_records(&json),
            Err(ParseError::TooManyRepetitions {
                index: 0,
                len: too_many,
                max: crate::stats::MAX_REPETITIONS,
            })
        );
    }

    #[test]
    fn parse_rejects_too_many_records() {
        // MAX_RECORDS + 1 minimal, otherwise-valid records: bounds the other factor a corrupt or
        // tampered results file could inflate, since repetitions per record are already capped.
        let one_record = r#"{
            "build_commit": "d",
            "build_number": 1,
            "model_filename": "m",
            "n_prompt": 64,
            "n_gen": 0,
            "n_depth": 0,
            "samples_ns": [1, 2, 3, 4],
            "samples_ts": [1.0, 2.0, 3.0, 4.0]
        }"#;
        let too_many = MAX_RECORDS + 1;
        let json = format!("[{}]", vec![one_record; too_many].join(","));
        assert_eq!(
            parse_records(&json),
            Err(ParseError::TooManyRecords { len: too_many, max: MAX_RECORDS })
        );
    }
}
