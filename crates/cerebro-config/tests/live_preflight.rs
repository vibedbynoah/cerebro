//! Settings route a real launch decision through `cerebro-supervisor`'s preflight.
//!
//! Machine-dependent: needs an official llama.cpp build (`CEREBRO_LLAMACPP_OFFICIAL`) and the tiny
//! GGUF (`CEREBRO_TINY_GGUF`). Preflight hashes and audits the build and binds a loopback port for
//! a moment; nothing is spawned and no model is loaded.

#![cfg(windows)]

use cerebro_config::{open_approved_model, KeyFile, LaunchRequest, Secret, Settings, Source};
use cerebro_supervisor::{preflight, PortRequest, PreflightWarning, Protection};

#[test]
#[ignore = "needs an official llama.cpp build (CEREBRO_LLAMACPP_OFFICIAL) and the tiny GGUF \
            (CEREBRO_TINY_GGUF); run explicitly"]
fn settings_protections_decide_a_real_preflight_both_ways() {
    let backend = cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|m| panic!("{m}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|m| panic!("{m}"));
    let (manifest, _) = cerebro_probe::backend::hash_directory(&backend).expect("hash the official build");

    let mut settings = Settings::default();
    let folder = model.parent().expect("the model has a folder").to_path_buf();
    settings
        .apply_patch(
            &serde_json::json!({
                "backend": { "dir": backend },
                "models": { "folders": [folder] },
            }),
            Source::Gui,
        )
        .expect("point the settings at the fixtures");
    let (model_path, _handle) = open_approved_model(&settings, &model).expect("the model is in a model folder");

    let keys = tempfile::tempdir().expect("tempdir");
    let key = KeyFile::create(keys.path(), &Secret::new(b"0123456789abcdef".repeat(4))).expect("key file");
    let request = |manifest| LaunchRequest {
        model_path: model_path.clone(),
        draft_model_path: None,
        manifest,
        port: PortRequest::Any,
        api_key_file: key.path().to_path_buf(),
    };

    // Every protection is on by default. On a given machine the first decision may be one of their
    // refusals: an elevated shell meets the elevation refusal, and a build unpacked under `C:\`
    // inherits `Authenticated Users: Modify` and meets the folder audit (the official build here
    // does). Each such refusal must name its setting, and turning that setting off must be the way
    // past it: the owner rule end to end. Anything else is a real failure.
    let mut turned_off = 0usize;
    let clean = loop {
        match preflight(&settings.launch_spec(request(manifest.clone()))) {
            Ok(done) => break done,
            Err(err) => {
                let key = match err.governing_protection() {
                    Some(Protection::ElevationRefusal) => "refuse_elevation",
                    Some(Protection::FolderPermissionAudit) => "audit_folder_permissions",
                    _ => panic!("an intact build must pass or name a machine-state setting: {err}"),
                };
                assert!(turned_off < 2, "the same refusal came back after its setting was off: {err}");
                settings
                    .apply_patch(&serde_json::json!({"protections": {key: false}}), Source::Gui)
                    .expect("turn the protection off");
                turned_off += 1;
            }
        }
    };
    assert_eq!(settings.disabled_protections().len(), turned_off);
    assert_eq!(clean.warnings().len(), turned_off, "{:?}", clean.warnings());
    let expected_warnings = turned_off;

    // A manifest that no longer matches the directory is refused, naming the setting.
    let mut tampered = manifest.clone();
    let first = tampered.keys().next().cloned().expect("the build has files");
    tampered.insert(first, "0".repeat(64));
    let err = preflight(&settings.launch_spec(request(tampered.clone()))).unwrap_err();
    assert_eq!(err.governing_protection(), Some(Protection::BackendIntegrityVerification), "{err}");

    // Turning the check off in settings is honoured, and recorded exactly once.
    settings
        .apply_patch(
            &serde_json::json!({"protections": {"verify_backend_integrity": false}}),
            Source::Gui,
        )
        .expect("turn integrity verification off");
    assert_eq!(
        settings.disabled_protections().iter().filter(|d| d.key == "backend_integrity_verification").count(),
        1
    );
    let allowed = preflight(&settings.launch_spec(request(tampered))).expect("the user turned the check off");
    let integrity_notes = allowed
        .warnings()
        .iter()
        .filter(|w| {
            matches!(w, PreflightWarning::ProtectionDisabled(p)
                if p.protection() == Protection::BackendIntegrityVerification)
        })
        .count();
    assert_eq!(integrity_notes, 1, "{:?}", allowed.warnings());
    assert_eq!(allowed.warnings().len(), expected_warnings + 1, "{:?}", allowed.warnings());
}
