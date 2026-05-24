use super::*;

#[test]
fn model_lifecycle_parser_is_exposed_from_cli_model_module() {
    let args = [
        "plan",
        "Qwen/Qwen3.6-35B-A3B",
        "--revision",
        "refs/pr/1",
        "--metadata-only",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();

    let request = llm_engine::cli::model::model_lifecycle_request_from_args("plan", &args)
        .expect("CLI lifecycle request parses");
    let options = request.resolve().expect("request defaults resolve");

    assert_eq!(options.repo_id.as_str(), "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(options.revision, "refs/pr/1");
    assert!(options.metadata_only);
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
    assert_eq!(
        value["cache_policy"]["namespace_fields"]
            .as_array()
            .expect("namespace fields")
            .iter()
            .map(|field| field.as_str().expect("namespace field"))
            .collect::<Vec<_>>(),
        [
            "model_id",
            "backend",
            "family",
            "quantization",
            "repo_id",
            "resolved_commit",
            "profile",
            "prompt_cache_key",
            "tool_schema",
            "request_mode",
            "cache_layout_version",
            "cache_capacity_bucket",
            "max_prefill_tokens",
        ]
    );
    assert!(
        value["cache_policy"]["benchmark_metrics"]
            .as_array()
            .expect("benchmark cache metrics")
            .contains(&Value::String("prefix_cache_hit_rate".to_owned()))
    );
    assert!(
        value["cache_policy"]["benchmark_metrics"]
            .as_array()
            .expect("benchmark cache metrics")
            .contains(&Value::String("prefix_cache_hit_tokens".to_owned()))
    );
    assert!(
        value["cache_policy"]["benchmark_metrics"]
            .as_array()
            .expect("benchmark cache metrics")
            .contains(&Value::String("prefix_cache_miss_tokens".to_owned()))
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
            "multi-turn-lifecycle",
            "same-long-prompt-twice",
            "shared-prefix-short-suffix-variation"
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
