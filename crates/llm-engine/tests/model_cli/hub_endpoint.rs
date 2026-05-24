use super::*;

#[test]
fn model_plan_hub_endpoint_flag_uses_configured_hub() {
    let (endpoint, server) = spawn_test_hub_server(|listener| {
        let (request, mut stream) = accept_hub_request(&listener);
        assert_model_info_request(&request);
        write_json_response(&mut stream, &hub_model_info_body());
    });

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "plan",
            "Qwen/Qwen3.6-35B-A3B",
            "--hub-endpoint",
            endpoint.as_str(),
        ])
        .env_remove("HF_TOKEN")
        .env_remove("LLM_HUB_ENDPOINT")
        .output()
        .expect("run model plan with configured hub endpoint");

    server.join().expect("test hub server exits");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("plan JSON");
    assert_eq!(value["repo_id"]["id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        value["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(value["files_to_download"][0]["path"], "config.json");
}

#[test]
fn model_plan_hub_endpoint_env_uses_configured_hub() {
    let (endpoint, server) = spawn_test_hub_server(|listener| {
        let (request, mut stream) = accept_hub_request(&listener);
        assert_model_info_request(&request);
        write_json_response(&mut stream, &hub_model_info_body());
    });

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "plan", "Qwen/Qwen3.6-35B-A3B"])
        .env("LLM_HUB_ENDPOINT", &endpoint)
        .env_remove("HF_TOKEN")
        .output()
        .expect("run model plan with env hub endpoint");

    server.join().expect("test hub server exits");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("plan JSON");
    assert_eq!(value["files_to_download"][0]["path"], "config.json");
}

#[test]
fn model_pull_hub_endpoint_flag_downloads_from_configured_hub() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (endpoint, server) = spawn_test_hub_server(|listener| {
        let (request, mut stream) = accept_hub_request(&listener);
        assert_model_info_request(&request);
        write_json_response(&mut stream, &hub_model_info_body());

        let (request, mut stream) = accept_hub_request(&listener);
        assert!(
            request.starts_with(
                "GET /Qwen/Qwen3.6-35B-A3B/resolve/0123456789abcdef0123456789abcdef01234567/config.json"
            ),
            "request: {request}"
        );
        write_bytes_response(&mut stream, b"{}");
    });

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "pull",
            "Qwen/Qwen3.6-35B-A3B",
            "--metadata-only",
            "--model-home",
        ])
        .arg(temp.path())
        .args(["--hub-endpoint", endpoint.as_str()])
        .env_remove("HF_TOKEN")
        .env_remove("LLM_HUB_ENDPOINT")
        .output()
        .expect("run model pull with configured hub endpoint");

    server.join().expect("test hub server exits");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("pull JSON");
    assert_eq!(value["files"], 1);
    assert_eq!(
        value["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
}

#[test]
fn model_plan_hub_endpoint_rejects_http_with_hf_token() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "plan",
            "Qwen/Qwen3.6-35B-A3B",
            "--hub-endpoint",
            "http://127.0.0.1:9",
        ])
        .env("HF_TOKEN", "hf_test_token")
        .env_remove("LLM_HUB_ENDPOINT")
        .output()
        .expect("run model plan with HTTP endpoint and token");

    assert!(!output.status.success(), "plan unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refusing to send HF_TOKEN to non-HTTPS hub endpoint"),
        "stderr: {stderr}"
    );
}

#[test]
fn model_plan_rejects_invalid_repo_before_hub_request() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "plan", "not-a-model"])
        .output()
        .expect("run model plan");

    assert!(!output.status.success(), "plan unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("repo id must be org/name"),
        "stderr: {stderr}"
    );
}
