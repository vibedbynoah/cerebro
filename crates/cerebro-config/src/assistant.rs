//! How the assistant answers: the effort dial (thinking budget and verification rigor, see
//! `docs/specs/effort-levels.md`) and the verbosity dial (brief answers first by default).
//!
//! Both are user-editable end to end: every level's instruction, budget and passes can be rewritten,
//! levels can be added or removed, and either default can be changed. Levels are keyed by a stable
//! id (`"low"`, `"brief"`, ...) rather than stored as an array, so a settings screen can patch one
//! level's instruction with a JSON Merge Patch on `/effort/levels/high/instruction` without resending
//! the rest; `rank` orders them on the dial.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{cap_echo, ConfigError};
use crate::load::{check_name, name_schema, unique_map};

/// Largest thinking budget a level may name, in tokens. Far above any current model's context; a
/// bound so a typo cannot ask for four billion.
pub const MAX_THINKING_TOKENS: u32 = 1 << 20;

/// Longest instruction a level may carry, in characters. Appended to every system prompt, so it is
/// bounded well below a context window.
pub const MAX_INSTRUCTION_CHARS: usize = 8192;

/// Most levels one dial may hold.
pub const MAX_LEVELS: usize = 32;

/// Most review passes a level may request.
pub const MAX_REVIEW_PASSES: u8 = 3;

/// Most independent self-consistency paths a level may request.
pub const MAX_CONSISTENCY_PATHS: u8 = 9;

/// How much the model may think before answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingBudget {
    /// No thinking: answer directly. Serialized as `"off"`.
    Off,
    /// At most this many thinking tokens, `1..=`[`MAX_THINKING_TOKENS`]. Serialized as the number.
    Tokens(u32),
    /// The largest budget the loaded model supports. Serialized as `"model_max"`.
    ModelMax,
}

impl Serialize for ThinkingBudget {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ThinkingBudget::Off => s.serialize_str("off"),
            ThinkingBudget::Tokens(n) => s.serialize_u32(*n),
            ThinkingBudget::ModelMax => s.serialize_str("model_max"),
        }
    }
}

impl<'de> Deserialize<'de> for ThinkingBudget {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Tokens(u64),
            Word(String),
        }
        match Raw::deserialize(d)? {
            Raw::Tokens(n) => match u32::try_from(n) {
                Ok(n) if (1..=MAX_THINKING_TOKENS).contains(&n) => Ok(ThinkingBudget::Tokens(n)),
                _ => Err(serde::de::Error::custom(format!(
                    "thinking budget {n} is outside 1..={MAX_THINKING_TOKENS}; use \"off\" for none"
                ))),
            },
            Raw::Word(w) if w == "off" => Ok(ThinkingBudget::Off),
            Raw::Word(w) if w == "model_max" => Ok(ThinkingBudget::ModelMax),
            Raw::Word(w) => Err(serde::de::Error::custom(format!(
                "thinking budget {:?} is not \"off\", \"model_max\" or a token count",
                cap_echo(&w)
            ))),
        }
    }
}

/// One level on the effort dial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffortLevel {
    /// Position on the dial; lower is less effort. Unique within the dial.
    pub rank: u32,
    /// Display name, e.g. `"High"`.
    pub name: String,
    /// Thinking budget, mapped to the backend's reasoning control where the model supports one.
    pub thinking_budget: ThinkingBudget,
    /// Verification instruction appended to the system prompt.
    pub instruction: String,
    /// Extra passes in which the draft is sent back with a review instruction and the revision
    /// replaces it, `0..=`[`MAX_REVIEW_PASSES`].
    pub review_passes: u8,
    /// Whether a final, independent check runs in a fresh context ("find any error in this
    /// answer").
    pub independent_check: bool,
    /// Independent answer paths for questions with a checkable answer, majority- or
    /// verifier-picked; `1` means a single path, `1..=`[`MAX_CONSISTENCY_PATHS`].
    pub consistency_paths: u8,
}

/// The effort dial: its levels and the default.
///
/// Like every other section, a missing key takes its default, so a hand-written
/// `"effort": {"default_level": "high"}` is a complete section. `levels` is one value: a file that
/// writes a `levels` object replaces the built-in levels with exactly the ones it lists (a JSON
/// Merge Patch, by contrast, merges into the existing levels).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffortSection {
    /// Level used when a message does not pick one. Must name an entry in `levels`.
    #[serde(default = "default_effort_level")]
    pub default_level: String,
    /// Tokens always reserved for the answer, separately from the thinking budget, so thinking can
    /// never consume the whole budget and return nothing (measured on this machine with
    /// `--reasoning on`). `1..=`[`MAX_THINKING_TOKENS`].
    #[serde(default = "default_answer_reserve")]
    pub answer_reserve_tokens: u32,
    /// Levels by stable id (`[a-z0-9_-]`, 1 to 32 characters). A repeated id is refused.
    #[serde(default = "default_effort_levels", deserialize_with = "unique_map")]
    pub levels: BTreeMap<String, EffortLevel>,
}

fn default_effort_level() -> String {
    EffortSection::default().default_level
}

fn default_answer_reserve() -> u32 {
    EffortSection::default().answer_reserve_tokens
}

fn default_effort_levels() -> BTreeMap<String, EffortLevel> {
    EffortSection::default().levels
}

impl EffortSection {
    /// The default level's settings.
    ///
    /// `None` only for a value that never passed [`crate::Settings::validate`], which refuses a
    /// `default_level` naming no level.
    pub fn default_settings(&self) -> Option<&EffortLevel> {
        self.levels.get(&self.default_level)
    }

    /// Levels in dial order (by `rank`).
    pub fn ordered(&self) -> Vec<(&str, &EffortLevel)> {
        let mut v: Vec<(&str, &EffortLevel)> =
            self.levels.iter().map(|(id, l)| (id.as_str(), l)).collect();
        v.sort_by_key(|(_, l)| l.rank);
        v
    }

    fn validate(&self) -> Result<(), ConfigError> {
        check_levels(
            "/effort",
            &self.default_level,
            self.levels
                .iter()
                .map(|(id, l)| (id.as_str(), l.rank, l.name.as_str(), l.instruction.as_str())),
        )?;
        if !(1..=MAX_THINKING_TOKENS).contains(&self.answer_reserve_tokens) {
            return Err(rejected(
                "/effort/answer_reserve_tokens".to_string(),
                format!("must be within 1..={MAX_THINKING_TOKENS}"),
            ));
        }
        for (id, level) in &self.levels {
            let at = format!("/effort/levels/{}", cap_echo(id));
            if level.review_passes > MAX_REVIEW_PASSES {
                return Err(rejected(
                    format!("{at}/review_passes"),
                    format!("must be at most {MAX_REVIEW_PASSES}"),
                ));
            }
            if !(1..=MAX_CONSISTENCY_PATHS).contains(&level.consistency_paths) {
                return Err(rejected(
                    format!("{at}/consistency_paths"),
                    format!("must be within 1..={MAX_CONSISTENCY_PATHS}"),
                ));
            }
        }
        Ok(())
    }
}

impl Default for EffortSection {
    /// The five levels of `docs/specs/effort-levels.md`, with its prompt drafts verbatim; Medium by
    /// default.
    fn default() -> Self {
        let level = |rank, name: &str, budget, instruction: &str, review, independent, paths| {
            EffortLevel {
                rank,
                name: name.to_string(),
                thinking_budget: budget,
                instruction: instruction.to_string(),
                review_passes: review,
                independent_check: independent,
                consistency_paths: paths,
            }
        };
        Self {
            default_level: "medium".to_string(),
            answer_reserve_tokens: 2048,
            levels: BTreeMap::from([
                (
                    "low".to_string(),
                    level(
                        0,
                        "Low",
                        ThinkingBudget::Off,
                        "Be fast and direct. Use what you already know. Only stop to double-check \
                         when an error would have real consequences, such as money, safety or \
                         commands that change files.",
                        0,
                        false,
                        1,
                    ),
                ),
                (
                    "medium".to_string(),
                    level(
                        1,
                        "Medium",
                        ThinkingBudget::Tokens(1024),
                        "Before answering, check the key facts, numbers and any code you write. If \
                         you are unsure about something important, say so in one short sentence.",
                        0,
                        false,
                        1,
                    ),
                ),
                (
                    "high".to_string(),
                    level(
                        2,
                        "High",
                        ThinkingBudget::Tokens(4096),
                        "Think the problem through before answering. Verify every claim, \
                         calculation and line of code you give. State your assumptions. After \
                         drafting, reread your answer once and fix any mistake.",
                        0,
                        false,
                        1,
                    ),
                ),
                (
                    "xhigh".to_string(),
                    level(
                        3,
                        "XHigh",
                        ThinkingBudget::Tokens(16384),
                        "Assume every part of your answer is wrong until you have checked it. \
                         Recompute every number, trace every code path line by line, consider edge \
                         cases, and verify each claim. After drafting, review the whole answer \
                         critically and correct it. Clearly mark anything you could not verify.",
                        1,
                        false,
                        1,
                    ),
                ),
                (
                    "max".to_string(),
                    level(
                        4,
                        "Max",
                        ThinkingBudget::ModelMax,
                        "Verify everything; leave nothing unchecked. Recompute, retrace and \
                         re-derive every part of your answer. When your draft is done, review it as \
                         a demanding expert who is trying to find a single flaw in any line, \
                         number, claim or assumption, and fix what you find. Only give the answer \
                         once it would survive that review. If something cannot be confirmed, say \
                         precisely what and why instead of hiding it.",
                        1,
                        true,
                        3,
                    ),
                ),
            ]),
        }
    }
}

/// One level on the verbosity dial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerbosityLevel {
    /// Position on the dial; lower is terser. Unique within the dial.
    pub rank: u32,
    /// Display name, e.g. `"Brief"`.
    pub name: String,
    /// Instruction appended to the system prompt.
    pub instruction: String,
}

/// The verbosity dial: its levels and the default (Brief).
///
/// A missing key takes its default, as in [`EffortSection`], with the same rule for `levels`: a
/// file that writes one replaces the built-in levels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerbositySection {
    /// Level used when a conversation does not pick one. Must name an entry in `levels`.
    #[serde(default = "default_verbosity_level")]
    pub default_level: String,
    /// Levels by stable id (`[a-z0-9_-]`, 1 to 32 characters). A repeated id is refused.
    #[serde(default = "default_verbosity_levels", deserialize_with = "unique_map")]
    pub levels: BTreeMap<String, VerbosityLevel>,
}

fn default_verbosity_level() -> String {
    VerbositySection::default().default_level
}

fn default_verbosity_levels() -> BTreeMap<String, VerbosityLevel> {
    VerbositySection::default().levels
}

impl VerbositySection {
    /// The default level's settings; `None` only for a value that never passed validation.
    pub fn default_settings(&self) -> Option<&VerbosityLevel> {
        self.levels.get(&self.default_level)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        check_levels(
            "/verbosity",
            &self.default_level,
            self.levels
                .iter()
                .map(|(id, l)| (id.as_str(), l.rank, l.name.as_str(), l.instruction.as_str())),
        )
    }
}

impl Default for VerbositySection {
    /// Brief, Standard and Detailed; Brief by default (owner requirement: low upfront verbosity).
    fn default() -> Self {
        let level = |rank, name: &str, instruction: &str| VerbosityLevel {
            rank,
            name: name.to_string(),
            instruction: instruction.to_string(),
        };
        Self {
            default_level: "brief".to_string(),
            levels: BTreeMap::from([
                (
                    "brief".to_string(),
                    level(
                        0,
                        "Brief",
                        "Lead with the answer in one or two sentences. Add only the detail needed \
                         to act on it, and offer to expand rather than expanding unasked.",
                    ),
                ),
                (
                    "standard".to_string(),
                    level(
                        1,
                        "Standard",
                        "Give the answer first, then the key reasoning or steps in a short \
                         paragraph or list. Skip background the user did not ask for.",
                    ),
                ),
                (
                    "detailed".to_string(),
                    level(
                        2,
                        "Detailed",
                        "Give the answer first, then a thorough explanation: reasoning, steps, \
                         alternatives and caveats. Use headings or lists where they help.",
                    ),
                ),
            ]),
        }
    }
}

/// Validate both dials. Called from [`crate::Settings::validate`].
pub(crate) fn validate(effort: &EffortSection, verbosity: &VerbositySection) -> Result<(), ConfigError> {
    effort.validate()?;
    verbosity.validate()
}

fn rejected(pointer: String, reason: String) -> ConfigError {
    ConfigError::SettingRejected { pointer, reason }
}

/// Whether `id` is a legal level id: 1 to 32 of `[a-z0-9_-]`.
fn is_level_id(id: &str) -> bool {
    (1..=32).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Checks both dials share: at least one level, at most [`MAX_LEVELS`], legal ids, names and
/// instruction lengths, unique ranks, and a default that names a level.
fn check_levels<'a>(
    section: &str,
    default_level: &str,
    levels: impl ExactSizeIterator<Item = (&'a str, u32, &'a str, &'a str)>,
) -> Result<(), ConfigError> {
    let count = levels.len();
    if count == 0 || count > MAX_LEVELS {
        return Err(rejected(
            format!("{section}/levels"),
            format!("must hold 1..={MAX_LEVELS} levels, found {count}"),
        ));
    }
    let mut ranks = std::collections::BTreeSet::new();
    let mut found_default = false;
    for (id, rank, name, instruction) in levels {
        let at = format!("{section}/levels/{}", cap_echo(id));
        if !is_level_id(id) {
            return Err(rejected(at, "a level id is 1 to 32 of a-z, 0-9, _ and -".to_string()));
        }
        check_name(name, &format!("{at}/name"))?;
        if instruction.chars().count() > MAX_INSTRUCTION_CHARS {
            return Err(rejected(
                format!("{at}/instruction"),
                format!("must be at most {MAX_INSTRUCTION_CHARS} characters"),
            ));
        }
        if !ranks.insert(rank) {
            return Err(rejected(format!("{at}/rank"), format!("rank {rank} is used twice")));
        }
        found_default |= id == default_level;
    }
    if !found_default {
        return Err(rejected(
            format!("{section}/default_level"),
            format!("no level named {:?}", cap_echo(default_level)),
        ));
    }
    Ok(())
}

const LEVEL_ID_PATTERN: &str = "^[a-z0-9_-]{1,32}$";

/// JSON Schema for [`EffortSection`].
pub(crate) fn effort_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "description": "A missing key takes its default; a levels object replaces the built-in levels.",
        "properties": {
            "default_level": { "type": "string", "pattern": LEVEL_ID_PATTERN, "default": default_effort_level() },
            "answer_reserve_tokens": { "type": "integer", "minimum": 1, "maximum": MAX_THINKING_TOKENS, "default": default_answer_reserve() },
            "levels": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": MAX_LEVELS,
                "propertyNames": { "pattern": LEVEL_ID_PATTERN },
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["rank", "name", "thinking_budget", "instruction", "review_passes",
                                 "independent_check", "consistency_paths"],
                    "properties": {
                        "rank": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
                        "name": name_schema(),
                        "thinking_budget": {
                            "oneOf": [
                                { "type": "string", "enum": ["off", "model_max"] },
                                { "type": "integer", "minimum": 1, "maximum": MAX_THINKING_TOKENS },
                            ]
                        },
                        "instruction": { "type": "string", "maxLength": MAX_INSTRUCTION_CHARS },
                        "review_passes": { "type": "integer", "minimum": 0, "maximum": MAX_REVIEW_PASSES },
                        "independent_check": { "type": "boolean" },
                        "consistency_paths": { "type": "integer", "minimum": 1, "maximum": MAX_CONSISTENCY_PATHS },
                    }
                }
            }
        }
    })
}

/// JSON Schema for [`VerbositySection`].
pub(crate) fn verbosity_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "description": "A missing key takes its default; a levels object replaces the built-in levels.",
        "properties": {
            "default_level": { "type": "string", "pattern": LEVEL_ID_PATTERN, "default": default_verbosity_level() },
            "levels": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": MAX_LEVELS,
                "propertyNames": { "pattern": LEVEL_ID_PATTERN },
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["rank", "name", "instruction"],
                    "properties": {
                        "rank": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
                        "name": name_schema(),
                        "instruction": { "type": "string", "maxLength": MAX_INSTRUCTION_CHARS },
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_dials_are_valid_and_match_the_spec() {
        let effort = EffortSection::default();
        let verbosity = VerbositySection::default();
        validate(&effort, &verbosity).expect("defaults validate");

        let order: Vec<&str> = effort.ordered().into_iter().map(|(id, _)| id).collect();
        assert_eq!(order, ["low", "medium", "high", "xhigh", "max"]);
        assert_eq!(effort.levels["low"].thinking_budget, ThinkingBudget::Off);
        assert_eq!(effort.levels["max"].thinking_budget, ThinkingBudget::ModelMax);
        // Spec mechanics: XHigh and Max add a review pass, Max adds an independent check and
        // defaults to three self-consistency paths.
        assert_eq!(effort.levels["high"].review_passes, 0);
        assert_eq!(effort.levels["xhigh"].review_passes, 1);
        assert!(effort.levels["max"].independent_check);
        assert!(!effort.levels["xhigh"].independent_check);
        assert_eq!(effort.levels["max"].consistency_paths, 3);
        assert!(effort.levels["low"].instruction.starts_with("Be fast and direct."));

        assert_eq!(verbosity.default_level, "brief");
        assert!(verbosity.default_settings().is_some());
    }

    #[test]
    fn a_default_naming_no_level_is_refused_for_both_dials() {
        let effort = EffortSection {
            default_level: "ultra".to_string(),
            ..EffortSection::default()
        };
        let err = validate(&effort, &VerbositySection::default()).unwrap_err();
        assert!(matches!(&err, ConfigError::SettingRejected { pointer, .. } if pointer == "/effort/default_level"), "{err:?}");

        let mut verbosity = VerbositySection::default();
        verbosity.levels.remove("brief");
        let err = validate(&EffortSection::default(), &verbosity).unwrap_err();
        assert!(matches!(&err, ConfigError::SettingRejected { pointer, .. } if pointer == "/verbosity/default_level"), "{err:?}");
    }

    #[test]
    fn a_user_level_can_be_added_and_made_the_default() {
        let mut effort = EffortSection::default();
        let mut custom = effort.levels["high"].clone();
        custom.rank = 10;
        custom.name = "Code review".to_string();
        custom.review_passes = 2;
        effort.levels.insert("code-review".to_string(), custom);
        effort.default_level = "code-review".to_string();
        validate(&effort, &VerbositySection::default()).expect("a user level is accepted");
        assert_eq!(effort.default_settings().map(|l| l.review_passes), Some(2));
    }

    #[test]
    fn bad_ids_duplicate_ranks_and_out_of_range_passes_are_refused() {
        let base = EffortSection::default();

        let mut bad_id = base.clone();
        let level = bad_id.levels["low"].clone();
        bad_id.levels.insert("Has Space".to_string(), EffortLevel { rank: 99, ..level });
        assert!(bad_id.validate().is_err());

        let mut dup_rank = base.clone();
        if let Some(l) = dup_rank.levels.get_mut("max") {
            l.rank = 0;
        }
        assert!(dup_rank.validate().is_err());

        let mut passes = base.clone();
        if let Some(l) = passes.levels.get_mut("max") {
            l.review_passes = MAX_REVIEW_PASSES + 1;
        }
        assert!(passes.validate().is_err());

        let mut paths = base.clone();
        if let Some(l) = paths.levels.get_mut("max") {
            l.consistency_paths = 0;
        }
        assert!(paths.validate().is_err());

        let mut long = base.clone();
        if let Some(l) = long.levels.get_mut("low") {
            l.instruction = "x".repeat(MAX_INSTRUCTION_CHARS + 1);
        }
        assert!(long.validate().is_err());

        let mut reserve = base;
        reserve.answer_reserve_tokens = 0;
        assert!(reserve.validate().is_err());
    }

    #[test]
    fn a_partial_effort_or_verbosity_section_takes_defaults_for_missing_fields() {
        let effort: EffortSection =
            serde_json::from_value(serde_json::json!({"default_level": "high"})).expect("partial effort parses");
        assert_eq!(effort.default_level, "high");
        assert_eq!(effort.answer_reserve_tokens, EffortSection::default().answer_reserve_tokens);
        assert_eq!(effort.levels, EffortSection::default().levels);
        let verbosity: VerbositySection = serde_json::from_value(serde_json::json!({})).expect("empty verbosity parses");
        assert_eq!(verbosity, VerbositySection::default());
        validate(&effort, &verbosity).expect("the partial sections validate");

        // Through the whole document, as a hand-edited file would arrive.
        let settings: crate::Settings = serde_json::from_value(serde_json::json!({
            "version": crate::SCHEMA_VERSION,
            "effort": {"default_level": "high"},
            "verbosity": {"default_level": "detailed"},
        }))
        .expect("a partial dial section is a valid file");
        settings.validate().expect("and validates");
        assert_eq!(settings.effort.default_settings().map(|l| l.name.as_str()), Some("High"));

        // A `levels` object replaces the built-in levels (documented): the default must then name
        // one of the levels written.
        let replaced: EffortSection = serde_json::from_value(serde_json::json!({
            "levels": {"only": {"rank": 0, "name": "Only", "thinking_budget": "off", "instruction": "",
                                 "review_passes": 0, "independent_check": false, "consistency_paths": 1}}
        }))
        .expect("parses");
        assert_eq!(replaced.levels.len(), 1);
        assert!(replaced.validate().is_err(), "default `medium` no longer names a level");
    }

    #[test]
    fn a_duplicate_level_key_is_refused_for_both_dials() {
        let level = r#"{"rank": 0, "name": "A", "thinking_budget": "off", "instruction": "",
                        "review_passes": 0, "independent_check": false, "consistency_paths": 1}"#;
        let effort = format!(r#"{{"default_level": "a", "levels": {{"a": {level}, "a": {level}}}}}"#);
        let err = serde_json::from_str::<EffortSection>(&effort).unwrap_err();
        assert!(err.to_string().contains("duplicate key"), "{err}");
        let verbosity = r#"{"levels": {"brief": {"rank": 0, "name": "B", "instruction": ""},
                                       "brief": {"rank": 1, "name": "C", "instruction": ""}}}"#;
        let err = serde_json::from_str::<VerbositySection>(verbosity).unwrap_err();
        assert!(err.to_string().contains("duplicate key"), "{err}");
    }

    #[test]
    fn thinking_budget_round_trips_its_three_shapes_and_refuses_the_rest() {
        for (value, budget) in [
            (serde_json::json!("off"), ThinkingBudget::Off),
            (serde_json::json!("model_max"), ThinkingBudget::ModelMax),
            (serde_json::json!(4096), ThinkingBudget::Tokens(4096)),
        ] {
            let parsed: ThinkingBudget = serde_json::from_value(value.clone()).expect("parses");
            assert_eq!(parsed, budget);
            assert_eq!(serde_json::to_value(parsed).expect("serializes"), value);
        }
        for bad in [
            serde_json::json!(0),
            serde_json::json!(MAX_THINKING_TOKENS + 1),
            serde_json::json!("lots"),
            serde_json::json!(-1),
            serde_json::json!(true),
        ] {
            assert!(serde_json::from_value::<ThinkingBudget>(bad.clone()).is_err(), "{bad} accepted");
        }
    }
}
