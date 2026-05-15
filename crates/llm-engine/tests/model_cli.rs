use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::Value;
#[cfg(feature = "bench")]
use serde_json::json;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn runnable_qwen_files() -> Vec<HubFile> {
    vec![
        HubFile::new("config.json", 2, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 2, Some("\"tok\"")),
        HubFile::new(
            "model.safetensors",
            4,
            Some("3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"),
        ),
    ]
}

async fn write_runnable_qwen_files(snapshot_path: &Path) {
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    tokio::fs::write(snapshot_path.join("model.safetensors"), b"data")
        .await
        .expect("weights");
}

#[test]
#[cfg(feature = "bench")]
fn long_context_bench_dry_run_defines_qwen_promotion_profiles() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("snapshot");
    let baseline = temp.path().join("baseline.json");
    let trace = temp.path().join("trace.json");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-long-context",
            "--dry-run",
            "--profile",
            "all",
            "--model",
            llm_engine::DEFAULT_MODEL_ID,
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
    assert_eq!(value["model"]["id"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(
        value["model"]["snapshot_path"],
        snapshot.display().to_string()
    );
    assert_eq!(value["baseline"]["path"], baseline.display().to_string());
    assert_eq!(value["trace_output_path"], trace.display().to_string());
    assert_eq!(value["hardware"]["os"], std::env::consts::OS);
    assert_eq!(value["hardware"]["arch"], std::env::consts::ARCH);
    assert_eq!(value["cache_policy"]["cache_layout"], "shared-prefix-v1");
    assert!(
        value["cache_policy"]["benchmark_metrics"]
            .as_array()
            .expect("benchmark cache metrics")
            .contains(&Value::String("prefix_cache_hit_rate".to_owned()))
    );
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 1, "lanes: {lanes:?}");
    assert_eq!(lanes[0]["name"], "primary");
    assert_eq!(lanes[0]["status"], "dry_run");
    assert_eq!(lanes[0]["model"]["id"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(
        lanes[0]["model"]["snapshot_path"],
        snapshot.display().to_string()
    );

    let profiles = value["profiles"].as_array().expect("profiles array");
    assert_eq!(profiles.len(), 3, "profiles: {profiles:?}");
    let promotion = profiles
        .iter()
        .find(|profile| profile["name"] == "qwen-135k-promotion")
        .expect("135K promotion profile");
    assert_eq!(promotion["target_tokens"], 135_000);
    assert_eq!(promotion["release_blocking"], true);
    let frontier = profiles
        .iter()
        .find(|profile| profile["name"] == "qwen-200k-characterization")
        .expect("200K characterization profile");
    assert_eq!(frontier["target_tokens"], 200_000);
    assert_eq!(frontier["release_blocking"], false);
    let max_context = profiles
        .iter()
        .find(|profile| profile["name"] == "qwen-256k-characterization")
        .expect("256K characterization profile");
    assert_eq!(max_context["target_tokens"], 256_000);
    assert_eq!(max_context["release_blocking"], false);

    let case_names = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .map(|case| case["name"].as_str().expect("case name"))
        .collect::<Vec<_>>();
    assert_eq!(
        case_names,
        [
            "plain-recall",
            "json-object-recall",
            "required-tool-recall",
            "streamed-required-tool-recall",
            "multi-turn-lifecycle"
        ]
    );
    let required_tool = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .find(|case| case["name"] == "required-tool-recall")
        .expect("required tool case");
    let streamed_tool = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .find(|case| case["name"] == "streamed-required-tool-recall")
        .expect("streamed required tool case");
    assert!(
        required_tool["response_contract"]
            .as_str()
            .expect("required tool response contract")
            .contains("tool_calls")
    );
    assert!(
        streamed_tool["response_contract"]
            .as_str()
            .expect("streamed tool response contract")
            .contains("tool_calls")
    );
}

#[test]
#[cfg(feature = "bench")]
fn long_context_bench_dry_run_accepts_named_backend_lanes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let native_snapshot = temp.path().join("native");
    let mlx_snapshot = temp.path().join("mlx");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-long-context",
            "--dry-run",
            "--profile",
            "135k",
            "--lane",
        ])
        .arg(format!(
            "name=native-metal,endpoint=http://127.0.0.1:3101,snapshot={},model=local-native",
            native_snapshot.display()
        ))
        .args(["--lane"])
        .arg(format!(
            "name=mlx,endpoint=http://127.0.0.1:3102,snapshot={},model=local-mlx",
            mlx_snapshot.display()
        ))
        .output()
        .expect("run qwen long-context lane dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 2, "lanes: {lanes:?}");
    assert_eq!(lanes[0]["name"], "native-metal");
    assert_eq!(lanes[0]["model"]["id"], "local-native");
    assert_eq!(lanes[0]["model"]["endpoint"], "http://127.0.0.1:3101");
    assert_eq!(
        lanes[0]["model"]["snapshot_path"],
        native_snapshot.display().to_string()
    );
    assert_eq!(lanes[1]["name"], "mlx");
    assert_eq!(lanes[1]["model"]["id"], "local-mlx");
    assert_eq!(value["model"]["id"], "local-native");
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_dry_run_records_template_model_and_phases() {
    let temp = tempfile::tempdir().expect("tempdir");
    let trace = temp.path().join("qwen-mlx-tool-normalized.json");
    let snapshot = temp.path().join("snapshot");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "2",
            "--samples",
            "3",
            "--context-tokens",
            "2048",
            "--concurrent-requests",
            "2",
            "--concurrent-samples",
            "1",
            "--output",
        ])
        .arg(&trace)
        .args(["--lane"])
        .arg(format!(
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,launched_model_id={},snapshot={},kind=direct_mlx,model_addressing=loaded_model_id,mlx_prompt_cache_size=4096,mlx_prompt_cache_bytes=unset,mlx_prefill_step_size=8192,mlx_prompt_concurrency=4,mlx_decode_concurrency=2",
            snapshot.display(),
            snapshot.display()
        ))
        .args(["--lane"])
        .arg(
            "name=proxy,endpoint=http://127.0.0.1:3000,model=qwen-proxy,kind=kir_ai_proxy,model_addressing=default_model,template=sidecar-chat-template-args,mlx_prompt_cache_size=default,mlx_prompt_cache_bytes=1073741824,mlx_prefill_step_size=default,mlx_prompt_concurrency=default,mlx_decode_concurrency=default",
        )
        .output()
        .expect("run qwen mlx tool normalized dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    let trace_value: Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("trace output file"))
            .expect("trace JSON output");

    assert_eq!(value["benchmark"], "qwen-mlx-tool-normalized");
    assert_eq!(trace_value["benchmark"], "qwen-mlx-tool-normalized");
    assert_eq!(value["status"], "dry_run");
    assert_eq!(value["warmups"], 2);
    assert_eq!(value["samples"], 3);
    assert_eq!(value["context_tokens"], 2048);
    assert_eq!(value["concurrent_requests"], 2);
    assert_eq!(value["concurrent_samples"], 1);
    assert_eq!(value["effective_concurrent_samples"], 1);
    assert!(
        value["repo_revision"]["commit_sha"]
            .as_str()
            .expect("repo commit sha")
            .len()
            >= 7
    );
    assert!(value["repo_revision"]["dirty"].is_boolean());
    assert_eq!(
        value["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|case| case.as_str().expect("case"))
            .collect::<Vec<_>>(),
        [
            "tool_required",
            "tool_required_stream",
            "json_object",
            "omp_repeated_prefix"
        ]
    );
    assert_eq!(
        value["schema_variants"]
            .as_array()
            .expect("schema variants")
            .iter()
            .map(|variant| variant.as_str().expect("schema variant"))
            .collect::<Vec<_>>(),
        [
            "baseline_current",
            "canonical_current",
            "baseline_permuted_equivalent",
            "canonical_permuted_equivalent"
        ]
    );
    assert_eq!(
        value["tool_choice_variants"]
            .as_array()
            .expect("tool choice variants")
            .iter()
            .map(|variant| variant.as_str().expect("tool choice variant"))
            .collect::<Vec<_>>(),
        ["required", "function"]
    );
    assert_eq!(
        value["cache_phases"]
            .as_array()
            .expect("cache phases")
            .iter()
            .map(|phase| phase.as_str().expect("phase"))
            .collect::<Vec<_>>(),
        ["cold", "warm_same_prompt", "warm_same_tool_schema"]
    );

    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 2, "lanes: {lanes:?}");
    assert_eq!(lanes[0]["name"], "direct");
    assert_eq!(lanes[0]["kind"], "direct_mlx");
    assert_eq!(lanes[0]["declared_model_id"], "qwen-loaded");
    assert_eq!(lanes[0]["effective_request_model_id"], "qwen-loaded");
    assert_eq!(
        lanes[0]["launched_model_id"],
        snapshot.display().to_string()
    );
    assert_eq!(lanes[0]["model_identity_source"], "lane_launched_model_id");
    assert_eq!(
        lanes[0]["snapshot_identity"]["id"],
        snapshot.display().to_string()
    );
    assert_eq!(lanes[0]["model_addressing"], "loaded_model_id");
    assert_eq!(lanes[0]["snapshot_path"], snapshot.display().to_string());
    assert_eq!(lanes[0]["mlx_lm_settings"]["mlx_prompt_cache_size"], 4096);
    assert_eq!(
        lanes[0]["mlx_lm_settings"]["mlx_prompt_cache_bytes"],
        "unset"
    );
    assert_eq!(lanes[0]["mlx_lm_settings"]["mlx_prefill_step_size"], 8192);
    assert_eq!(lanes[0]["mlx_lm_settings"]["mlx_prompt_concurrency"], 4);
    assert_eq!(lanes[0]["mlx_lm_settings"]["mlx_decode_concurrency"], 2);
    assert_eq!(
        lanes[0]["qwen_thinking_policy"]["template"],
        "qwen-no-thinking"
    );
    assert_eq!(
        lanes[0]["qwen_thinking_policy"]["request_chat_template_kwargs"]["enable_thinking"],
        false
    );

    assert_eq!(lanes[1]["name"], "proxy");
    assert_eq!(lanes[1]["kind"], "kir_ai_proxy");
    assert_eq!(lanes[1]["declared_model_id"], "qwen-proxy");
    assert_eq!(
        lanes[1]["effective_request_model_id"],
        llm_engine::DEFAULT_MODEL_ID
    );
    assert_eq!(lanes[1]["model_addressing"], "default_model");
    assert_eq!(
        lanes[1]["qwen_thinking_policy"]["template"],
        "sidecar-chat-template-args"
    );
    assert!(
        lanes[1]["qwen_thinking_policy"]
            .get("request_chat_template_kwargs")
            .is_none()
    );
    assert_eq!(
        lanes[1]["mlx_lm_settings"]["mlx_prompt_cache_size"],
        "default"
    );
    assert_eq!(
        lanes[1]["mlx_lm_settings"]["mlx_prompt_cache_bytes"],
        1073741824
    );

    let planned = lanes[0]["samples"].as_array().expect("planned samples");
    assert_eq!(planned.len(), 225);
    assert!(planned.iter().all(|sample| sample["status"] == "dry_run"));
    assert!(
        planned
            .iter()
            .all(|sample| sample["run_mode"] == "sequential")
    );
    assert!(
        planned
            .iter()
            .all(|sample| sample["planned_prompt_tokens"] == 2048)
    );
    assert_eq!(planned[0]["case"], "tool_required");
    assert_eq!(planned[0]["cache_phase"], "cold");
    assert_eq!(planned[0]["schema_variant"], "baseline_current");
    assert_eq!(planned[0]["tool_choice_variant"], "required");
    assert_eq!(planned[0]["schema_canonicalized"], false);
    assert_eq!(planned[0]["schema_permuted"], false);
    assert_eq!(planned[0]["prewarmed"], false);
    assert!(planned[0]["tool_schema_sha256"].as_str().is_some());
    assert!(planned[0]["tool_schema_bytes"].as_u64().is_some());
    assert_eq!(planned[0]["sample_index"], 0);
    assert_eq!(planned[3]["cache_phase"], "warm_same_prompt");
    assert_eq!(planned[3]["prewarmed"], true);
    assert_eq!(planned[6]["cache_phase"], "warm_same_tool_schema");
    assert_eq!(planned[6]["prewarmed"], true);
    let canonical_current = planned
        .iter()
        .find(|sample| {
            sample["case"] == "tool_required"
                && sample["schema_variant"] == "canonical_current"
                && sample["tool_choice_variant"] == "required"
                && sample["cache_phase"] == "cold"
                && sample["sample_index"] == 0
        })
        .expect("canonical current sample");
    let canonical_permuted = planned
        .iter()
        .find(|sample| {
            sample["case"] == "tool_required"
                && sample["schema_variant"] == "canonical_permuted_equivalent"
                && sample["tool_choice_variant"] == "required"
                && sample["cache_phase"] == "cold"
                && sample["sample_index"] == 0
        })
        .expect("canonical permuted sample");
    assert_eq!(
        canonical_current["tool_schema_sha256"],
        canonical_permuted["tool_schema_sha256"]
    );
    let json_control = planned
        .iter()
        .find(|sample| sample["case"] == "json_object")
        .expect("json control sample");
    assert_eq!(json_control["schema_variant"], "none");
    assert_eq!(json_control["tool_choice_variant"], "none");
    assert!(json_control.get("tool_schema_sha256").is_none());

    let concurrent = lanes[0]["concurrent_samples"]
        .as_array()
        .expect("planned concurrent samples");
    assert_eq!(concurrent.len(), 150);
    assert!(
        concurrent
            .iter()
            .all(|sample| sample["run_mode"] == "concurrent")
    );
    assert_eq!(concurrent[0]["sample_index"], 0);
    assert_eq!(concurrent[0]["request_index"], 0);
    assert_eq!(concurrent[1]["sample_index"], 0);
    assert_eq!(concurrent[1]["request_index"], 1);
    assert!(
        concurrent
            .iter()
            .any(|sample| sample["case"] == "omp_repeated_prefix")
    );

    let summary = value["summary"].as_array().expect("summary rows");
    assert!(
        summary.iter().any(|row| {
            row["lane"] == "direct"
                && row["case"] == "omp_repeated_prefix"
                && row["schema_variant"] == "canonical_permuted_equivalent"
                && row["tool_choice_variant"] == "function"
                && row["cache_phase"] == "cold"
                && row["run_mode"] == "concurrent"
        }),
        "summary rows: {summary:?}"
    );
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_cache_prefill_profile_dry_run_emits_sweep_matrix() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    std::fs::create_dir_all(&snapshot).expect("raw HF snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--sweep-profile",
            "qwen-mlx-cache-prefill",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--concurrent-requests",
            "2",
            "--concurrent-samples",
            "1",
            "--snapshot",
        ])
        .arg(&snapshot)
        .output()
        .expect("run qwen mlx cache prefill profile dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["benchmark"], "qwen-mlx-tool-normalized");
    assert_eq!(value["status"], "dry_run");
    assert_eq!(value["sweep_profile"], "qwen-mlx-cache-prefill");
    assert!(
        value["repo_revision"]["commit_sha"]
            .as_str()
            .expect("repo commit sha")
            .len()
            >= 7
    );

    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane["name"].as_str().expect("lane name"))
            .collect::<Vec<_>>(),
        [
            "mlx-default",
            "mlx-cache-size-4096",
            "mlx-cache-bytes-1g",
            "mlx-prefill-2048",
            "mlx-prefill-4096",
            "mlx-prefill-8192",
            "mlx-concurrent-4x2",
            "kir-proxy",
        ]
    );
    assert_eq!(lanes[0]["endpoint"], "http://127.0.0.1:8080/v1");
    assert_eq!(lanes[7]["endpoint"], "http://127.0.0.1:3000");
    assert_eq!(
        lanes[0]["declared_model_id"],
        snapshot.display().to_string()
    );
    assert_eq!(
        lanes[0]["effective_request_model_id"],
        snapshot.display().to_string()
    );
    assert_eq!(lanes[0]["model_addressing"], "server_default");
    assert_eq!(
        lanes[0]["launched_model_id"],
        snapshot.display().to_string()
    );
    assert_eq!(
        lanes[0]["snapshot_identity"]["repo_id"],
        "mlx-community/Qwen3.6-35B-A3B-4bit"
    );
    assert_eq!(lanes[7]["kind"], "kir_ai_proxy");
    assert_eq!(
        lanes[7]["effective_request_model_id"],
        llm_engine::DEFAULT_MODEL_ID
    );
    assert_eq!(lanes[6]["mlx_lm_settings"]["mlx_prompt_concurrency"], 4);
    assert_eq!(lanes[6]["mlx_lm_settings"]["mlx_decode_concurrency"], 2);
    assert_eq!(lanes[0]["samples"].as_array().expect("samples").len(), 75);
    assert_eq!(
        lanes[0]["concurrent_samples"]
            .as_array()
            .expect("concurrent samples")
            .len(),
        150
    );
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_prefill_135k_profile_dry_run_defaults_to_prefill_suite() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    std::fs::create_dir_all(&snapshot).expect("raw HF snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--sweep-profile",
            "qwen-mlx-prefill-135k",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--snapshot",
        ])
        .arg(&snapshot)
        .output()
        .expect("run qwen mlx prefill 135k profile dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["sweep_profile"], "qwen-mlx-prefill-135k");
    assert_eq!(value["probe_suite"], "prefill_sweep_135k");
    assert_eq!(value["prefill_sweep"]["status"], "dry_run");
    assert_eq!(
        value["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|case| case.as_str().expect("case"))
            .collect::<Vec<_>>(),
        [
            "chat_stream",
            "tool_required_stream",
            "context_recall_stream_135k",
            "warm_prefix_repeated_turn_stream"
        ]
    );
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 12);
    assert_eq!(lanes[0]["name"], "mlx-prefill-default");
    assert_eq!(lanes[1]["name"], "kir-prefill-default");
    assert_eq!(lanes[10]["endpoint"], "http://127.0.0.1:8085/v1");
    assert_eq!(lanes[11]["endpoint"], "http://127.0.0.1:3005");
    assert_eq!(lanes[2]["mlx_lm_settings"]["mlx_prefill_step_size"], 512);
    assert_eq!(lanes[0]["samples"].as_array().expect("samples").len(), 12);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_uses_kir_ai_checkout_when_run_from_harness_repo() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&harness_repo)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from harness repo");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_accepts_exported_source_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_COMMIT", &source_sha)
        .env("LLM_ENGINE_BENCH_REPO_BRANCH", "main")
        .env("LLM_ENGINE_BENCH_REPO_DIRTY", "false")
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from exported workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
    assert_eq!(value["repo_revision"]["branch"], "main");
    assert_eq!(value["repo_revision"]["dirty"], false);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_reads_kir_ai_origin_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");
    std::fs::write(
        exported_workspace.join(".kir-ai-origin.json"),
        json!({
            "repo_revision": {
                "commit_sha": source_sha,
                "branch": "main",
                "dirty": false
            }
        })
        .to_string(),
    )
    .expect("origin metadata");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from origin metadata workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
    assert_eq!(value["repo_revision"]["branch"], "main");
    assert_eq!(value["repo_revision"]["dirty"], false);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_does_not_walk_into_parent_harness_repo() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from exported workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
}

#[test]
fn serve_help_prints_without_backend_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--help"])
        .output()
        .expect("run serve help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--addr"), "stdout: {stdout}");
    assert!(stdout.contains("--snapshot"), "stdout: {stdout}");
    assert!(stdout.contains("--snapshot-alias"), "stdout: {stdout}");
    assert!(stdout.contains("--model-alias"), "stdout: {stdout}");
    assert!(stdout.contains("--model-id"), "stdout: {stdout}");
    assert!(
        stdout.contains("--loader <native-metal|mlx>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("--backend <native-metal|mlx>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("--family <qwen|deep_seek|gemma|llama>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("Raw native snapshots infer Qwen/Gemma from config.json"),
        "stdout: {stdout}"
    );
    assert!(
        !stdout.contains("--protocol-test-backend"),
        "serve help should not advertise the hardcoded protocol backend as a normal serving option: {stdout}"
    );
    assert!(stdout.contains("--max-new-tokens <n>"), "stdout: {stdout}");
    assert!(
        stdout.contains("[default: 256]"),
        "stdout should document the usable native generation default: {stdout}"
    );
    assert!(
        stdout.contains("--max-prefill-tokens <n>"),
        "stdout should document native prefill chunk sizing: {stdout}"
    );
    assert!(
        stdout.contains("[default: 2048"),
        "stdout should document the long-context native prefill default: {stdout}"
    );
    assert!(
        stdout.contains("memory-constrained correctness probes"),
        "stdout should make low native prefill chunks an explicit probe-only override: {stdout}"
    );
    assert!(
        stdout.contains("--max-json-body-bytes <bytes>"),
        "serve help should document configurable JSON body limits: {stdout}"
    );
    assert!(
        stdout.contains("--max-message-content-bytes <bytes>"),
        "serve help should document configurable chat message limits: {stdout}"
    );
    assert!(
        stdout.contains("--max-completion-prompt-bytes <bytes>"),
        "serve help should document configurable completion prompt limits: {stdout}"
    );
    assert!(
        stdout.contains("--canonical-tool-schemas"),
        "serve help should document production opt-in tool schema canonicalization: {stdout}"
    );
    assert!(
        stdout.contains("LLM_ENGINE_CANONICAL_TOOL_SCHEMAS"),
        "serve help should document the canonicalization environment variable: {stdout}"
    );
    assert!(
        stdout.contains("--mlx-stream-usage <true|false>"),
        "serve help should document MLX streaming usage forwarding: {stdout}"
    );
    assert!(
        stdout.contains("LLM_ENGINE_MLX_STREAM_USAGE"),
        "serve help should document the MLX streaming usage environment variable: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("requires --snapshot"),
        "stderr unexpectedly validated backend: {stderr}"
    );
}

#[tokio::test]
async fn serve_without_snapshot_requires_explicit_backend() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if child.try_wait().expect("poll serve").is_some() {
            let output = child.wait_with_output().expect("collect serve output");
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("--snapshot"), "stderr: {stderr}");
            assert!(
                !stderr.contains("--protocol-test-backend"),
                "missing snapshot error should not suggest a hardcoded backend for real serving: {stderr}"
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    child.kill().expect("kill hanging serve");
    let _ = child.wait();
    panic!("serve bound the protocol test backend instead of failing without --snapshot");
}

#[test]
#[cfg(not(feature = "bench"))]
fn bench_command_without_feature_errors_clearly() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .arg("bench")
        .output()
        .expect("run llm-engine bench without bench feature");

    assert!(
        !output.status.success(),
        "bench command unexpectedly succeeded without bench feature"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires the llm-engine `bench` feature"),
        "stderr did not explain missing bench feature: {stderr}"
    );
}

#[test]
#[cfg(not(feature = "diagnostics"))]
fn diagnostics_command_without_feature_errors_clearly() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "inspect", "/tmp/missing-snapshot"])
        .output()
        .expect("run llm-engine model inspect without diagnostics feature");

    assert!(
        !output.status.success(),
        "diagnostics command unexpectedly succeeded without diagnostics feature"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires the llm-engine `diagnostics` feature"),
        "stderr did not explain missing diagnostics feature: {stderr}"
    );
}

#[tokio::test]
async fn serve_with_missing_snapshot_alias_fails_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--snapshot-alias",
            "missing-model",
            "--loader",
            "mlx",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run serve with missing snapshot alias");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model alias `missing-model`"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_raw_mlx_snapshot_without_family_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("raw-mlx");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0", "--snapshot"])
        .arg(&snapshot)
        .args(["--loader", "mlx"])
        .output()
        .expect("run serve with manifestless MLX snapshot");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("snapshot loader `mlx` requires model family metadata"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_deepseek_native_metal_family_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("raw-deepseek");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0", "--snapshot"])
        .arg(&snapshot)
        .args(["--loader", "native-metal", "--family", "deep_seek"])
        .output()
        .expect("run serve with unsupported native DeepSeek family");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("snapshot loader `native-metal` is not supported for family `deep_seek`"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_protocol_test_backend_requires_explicit_fixture_ack() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--protocol-test-backend",
            "--hub-endpoint",
            "not a url",
        ])
        .output()
        .expect("run serve with protocol backend but no acknowledgement");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--i-understand-this-is-not-real-inference"),
        "stderr: {stderr}"
    );
    assert!(
        !stderr.contains("invalid hub endpoint"),
        "protocol-test acknowledgement must be checked before unrelated config: {stderr}"
    );
}

#[tokio::test]
async fn serve_legacy_deterministic_test_backend_alias_requires_explicit_fixture_ack() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--deterministic-test-backend",
            "--hub-endpoint",
            "not a url",
        ])
        .output()
        .expect("run serve with legacy deterministic backend alias but no acknowledgement");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--deterministic-test-backend"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("--i-understand-this-is-not-real-inference"),
        "stderr: {stderr}"
    );
    assert!(
        !stderr.contains("requires --snapshot"),
        "legacy deterministic backend alias must be recognized before snapshot validation: {stderr}"
    );
    assert!(
        !stderr.contains("invalid hub endpoint"),
        "protocol-test acknowledgement must be checked before unrelated config: {stderr}"
    );
}

#[tokio::test]
#[cfg(feature = "test-utils")]
async fn serve_legacy_deterministic_test_backend_alias_accepts_explicit_fixture_ack() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--deterministic-test-backend",
            "--i-understand-this-is-not-real-inference",
            "--hub-endpoint",
            "not a url",
        ])
        .output()
        .expect("run serve with acknowledged legacy deterministic backend alias");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid hub endpoint"), "stderr: {stderr}");
    assert!(
        !stderr.contains("requires --snapshot"),
        "legacy deterministic backend alias should reach protocol backend setup: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[tokio::test]
#[cfg(feature = "test-utils")]
async fn serve_rejects_invalid_hub_endpoint_without_panic() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--protocol-test-backend",
            "--i-understand-this-is-not-real-inference",
            "--hub-endpoint",
            "not a url",
        ])
        .output()
        .expect("run serve with invalid hub endpoint");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid hub endpoint"), "stderr: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[tokio::test]
async fn model_list_outputs_promoted_snapshots() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "list", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model list");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["snapshots"][0]["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        value["snapshots"][0]["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(
        value["snapshots"][0]["manifest_digest"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
}

#[tokio::test]
#[cfg(feature = "diagnostics")]
async fn model_inspect_outputs_snapshot_manifest_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "inspect"])
        .arg(&snapshot_path)
        .output()
        .expect("run model inspect");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["status"], "ready");
    assert_eq!(value["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(value["files"], 3);
    assert_eq!(value["total_bytes"], 8);
}

#[tokio::test]
async fn model_verify_outputs_snapshot_integrity_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "verify"])
        .arg(&snapshot_path)
        .output()
        .expect("run model verify");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["status"], "ok");
    assert_eq!(value["verified_files"], 3);
    assert_eq!(value["verified_bytes"], 8);
}

#[tokio::test]
async fn model_prune_dry_run_outputs_snapshot_usage_without_deleting() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "prune", "--dry-run", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model prune dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["dry_run"], true);
    assert_eq!(value["snapshots"], 1);
    assert_eq!(value["total_bytes"], 2);
    assert_eq!(value["reclaimable_bytes"], 0);
    assert!(snapshot_path.join("config.json").is_file());
}

#[tokio::test]
async fn model_prune_confirm_deletes_same_candidates_as_dry_run() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "abcdefabcdefabcdefabcdefabcdefabcdefabcd",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    let old = chrono::DateTime::parse_from_rfc3339("2026-04-01T00:00:00Z")
        .expect("fixed time")
        .with_timezone(&chrono::Utc);
    ModelStore::mark_snapshot_used_at(&snapshot_path, old)
        .await
        .expect("usage recorded");

    let dry_run = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "prune",
            "--dry-run",
            "--older-than-days",
            "7",
            "--keep-min-per-profile",
            "0",
            "--now",
            "2026-05-08T00:00:00Z",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run model prune dry-run");

    assert!(
        dry_run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let dry_run_value: Value = serde_json::from_slice(&dry_run.stdout).expect("json output");
    let dry_run_candidates = dry_run_value["candidates"]
        .as_array()
        .expect("candidate array");
    assert_eq!(dry_run_value["dry_run"], true);
    assert_eq!(dry_run_candidates.len(), 1);
    assert_eq!(
        dry_run_candidates[0]["path"],
        snapshot_path.display().to_string()
    );
    assert!(snapshot_path.join("config.json").is_file());

    let destructive = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "prune",
            "--confirm-delete",
            "--older-than-days",
            "7",
            "--keep-min-per-profile",
            "0",
            "--now",
            "2026-05-08T00:00:00Z",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run model prune destructive");

    assert!(
        destructive.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&destructive.stderr)
    );
    let destructive_value: Value =
        serde_json::from_slice(&destructive.stdout).expect("json output");
    let destructive_candidates = destructive_value["candidates"]
        .as_array()
        .expect("candidate array");
    assert_eq!(destructive_value["dry_run"], false);
    assert_eq!(destructive_candidates.len(), dry_run_candidates.len());
    assert_eq!(
        destructive_candidates[0]["path"],
        dry_run_candidates[0]["path"]
    );
    assert_eq!(
        destructive_value["deleted"]
            .as_array()
            .expect("deleted")
            .len(),
        1
    );
    assert!(!snapshot_path.exists());
}

#[tokio::test]
#[cfg(feature = "diagnostics")]
async fn model_list_and_inspect_show_quarantined_snapshots() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    let quarantined = store
        .quarantine_snapshot(&snapshot_path, "test corruption")
        .await
        .expect("quarantined");

    let list = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "list", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model list");

    assert!(
        list.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&list.stderr)
    );
    let list_value: Value = serde_json::from_slice(&list.stdout).expect("json output");
    assert_eq!(
        list_value["snapshots"].as_array().expect("snapshots").len(),
        0
    );
    assert_eq!(
        list_value["quarantined_snapshots"]
            .as_array()
            .expect("quarantined")
            .len(),
        1
    );
    assert_eq!(
        list_value["quarantined_snapshots"][0]["path"],
        quarantined.path.display().to_string()
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "inspect"])
        .arg(&quarantined.path)
        .output()
        .expect("run model inspect");

    assert!(
        inspect.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let inspect_value: Value = serde_json::from_slice(&inspect.stdout).expect("json output");
    assert_eq!(inspect_value["status"], "quarantined");
    assert_eq!(
        inspect_value["original_path"],
        snapshot_path.display().to_string()
    );
}
