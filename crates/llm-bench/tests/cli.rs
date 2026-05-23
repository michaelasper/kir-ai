use serde_json::Value;
use std::process::Command;

#[test]
fn qwen_long_context_dry_run_binary_defines_qwen_promotion_profiles() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("snapshot");
    let baseline = temp.path().join("baseline.json");
    let trace = temp.path().join("trace.json");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-bench"))
        .args([
            "qwen-long-context",
            "--dry-run",
            "--profile",
            "all",
            "--model",
            llm_bench::DEFAULT_MODEL_ID,
            "--snapshot",
        ])
        .arg(&snapshot)
        .args(["--baseline"])
        .arg(&baseline)
        .args(["--output"])
        .arg(&trace)
        .output()
        .expect("run qwen long-context bench dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    let trace_value: Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("trace output file"))
            .expect("trace JSON output");
    assert_eq!(trace_value["gate"], "qwen-long-context");
    assert_eq!(value["gate"], "qwen-long-context");
    assert_eq!(value["status"], "dry_run");
    assert_eq!(value["model"]["id"], llm_bench::DEFAULT_MODEL_ID);
    assert_eq!(
        value["model"]["snapshot_path"],
        snapshot.display().to_string()
    );
    assert_eq!(value["baseline"]["path"], baseline.display().to_string());
    assert_eq!(value["trace_output_path"], trace.display().to_string());

    let profiles = value["profiles"].as_array().expect("profiles array");
    assert_eq!(profiles.len(), 3, "profiles: {profiles:?}");
    assert!(
        profiles
            .iter()
            .any(|profile| profile["name"] == "qwen-135k-promotion")
    );
    assert!(
        profiles
            .iter()
            .any(|profile| profile["name"] == "qwen-200k-characterization")
    );
    assert!(
        profiles
            .iter()
            .any(|profile| profile["name"] == "qwen-256k-characterization")
    );
}
