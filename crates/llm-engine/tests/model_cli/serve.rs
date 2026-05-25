use super::*;

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
    assert!(
        stdout.contains("--snapshot-readiness <fast|deep>"),
        "stdout: {stdout}"
    );
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
        stdout.contains("--tls-cert <path>"),
        "serve help should document HTTPS certificate configuration: {stdout}"
    );
    assert!(
        stdout.contains("--tls-key <path>"),
        "serve help should document HTTPS private key configuration: {stdout}"
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
        stdout.contains("--log-level <level>"),
        "serve help should document startup log-level override: {stdout}"
    );
    assert!(
        stdout.contains("RUST_LOG"),
        "serve help should document RUST_LOG support: {stdout}"
    );
    assert!(
        stdout.contains("--native-prefix-cache-bytes <bytes>"),
        "serve help should document native prefix cache sizing: {stdout}"
    );
    assert!(
        stdout.contains("LLM_ENGINE_PREFIX_CACHE_BYTES"),
        "serve help should document the native prefix cache environment variable: {stdout}"
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
        stdout.contains("--max-tool-schema-depth <depth>"),
        "serve help should document configurable tool schema depth limits: {stdout}"
    );
    assert!(
        stdout.contains("--max-public-inference-requests-per-second <n>"),
        "serve help should document configurable public inference rate limits: {stdout}"
    );
    assert!(
        stdout.contains("--stream-stall-timeout <secs>"),
        "serve help should document configurable stream stall timeout: {stdout}"
    );
    assert!(
        stdout.contains("loopback without one generates a temporary token"),
        "serve help should document generated loopback admin token behavior: {stdout}"
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

#[test]
fn serve_rejects_invalid_rust_log_before_backend_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .env("RUST_LOG", "[invalid")
        .args(["serve", "--addr", "127.0.0.1:0"])
        .output()
        .expect("run serve with invalid RUST_LOG");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RUST_LOG"), "stderr: {stderr}");
    assert!(
        !stderr.contains("requires --snapshot"),
        "RUST_LOG validation should happen before backend validation: {stderr}"
    );
}

#[test]
fn serve_log_level_flag_takes_precedence_over_rust_log() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .env("RUST_LOG", "info")
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--log-level",
            "definitely-not-a-level",
        ])
        .output()
        .expect("run serve with invalid --log-level");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--log-level"), "stderr: {stderr}");
    assert!(
        !stderr.contains("requires --snapshot"),
        "--log-level validation should happen before backend validation: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_startup_numeric_config_outside_supported_ranges() {
    assert_invalid_serve_config(
        &["--max-concurrent-requests", "0"],
        "--max-concurrent-requests",
    );
    assert_invalid_serve_config(
        &["--max-concurrent-requests", "257"],
        "--max-concurrent-requests",
    );

    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("raw-native");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("snapshot dir");
    let snapshot = snapshot.to_string_lossy().into_owned();
    assert_invalid_serve_config(
        &[
            "--snapshot",
            snapshot.as_str(),
            "--loader",
            "native-metal",
            "--family",
            "qwen",
            "--max-new-tokens",
            "0",
        ],
        "--max-new-tokens",
    );
    assert_invalid_serve_config(
        &[
            "--snapshot",
            snapshot.as_str(),
            "--loader",
            "native-metal",
            "--family",
            "qwen",
            "--max-new-tokens",
            "65537",
        ],
        "--max-new-tokens",
    );
    assert_invalid_serve_config(
        &[
            "--snapshot",
            snapshot.as_str(),
            "--loader",
            "native-metal",
            "--family",
            "qwen",
            "--max-prefill-tokens",
            "0",
        ],
        "--max-prefill-tokens",
    );
    assert_invalid_serve_config(
        &[
            "--snapshot",
            snapshot.as_str(),
            "--loader",
            "native-metal",
            "--family",
            "qwen",
            "--max-prefill-tokens",
            "262145",
        ],
        "--max-prefill-tokens",
    );
}

fn assert_invalid_serve_config(extra_args: &[&str], expected_flag: &str) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_llm-engine"));
    command.args(["serve", "--addr", "127.0.0.1:0"]);
    command.args(extra_args);
    let output = command.output().expect("run serve with invalid config");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected_flag),
        "stderr should mention {expected_flag}: {stderr}"
    );
    assert!(
        !stderr.contains("requires --snapshot"),
        "startup config validation should happen before backend validation: {stderr}"
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

#[tokio::test]
async fn serve_tls_cert_requires_tls_key_before_backend_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--tls-cert",
            "/tmp/cert.pem",
        ])
        .output()
        .expect("run serve with incomplete TLS config");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--tls-key"), "stderr: {stderr}");
    assert!(
        !stderr.contains("requires --snapshot"),
        "TLS option validation should happen before backend validation: {stderr}"
    );
}

#[test]
#[cfg(feature = "test-utils")]
fn serve_loopback_without_admin_token_rejects_unauthenticated_admin_requests() {
    let addr = reserve_loopback_addr();
    let mut server = spawn_protocol_test_server(&addr, None);
    wait_for_server(&mut server.child, &addr);

    let response = http_get(&addr, "/admin/metrics", None).expect("admin metrics response");

    assert!(
        response.starts_with("HTTP/1.1 401"),
        "unauthenticated admin request should fail closed: {response}"
    );
}

#[test]
#[cfg(feature = "test-utils")]
fn serve_loopback_with_explicit_admin_token_accepts_bearer_token() {
    let addr = reserve_loopback_addr();
    let mut server = spawn_protocol_test_server(&addr, Some("secret-admin-token"));
    wait_for_server(&mut server.child, &addr);

    let missing_token =
        http_get(&addr, "/admin/metrics", None).expect("admin metrics response without token");
    assert!(
        missing_token.starts_with("HTTP/1.1 401"),
        "admin request without bearer token should fail: {missing_token}"
    );

    let authorized = http_get(
        &addr,
        "/admin/metrics",
        Some("Authorization: Bearer secret-admin-token\r\n"),
    )
    .expect("authorized admin metrics response");
    assert!(
        authorized.starts_with("HTTP/1.1 200"),
        "admin request with explicit bearer token should pass: {authorized}"
    );
}

#[test]
#[cfg(all(unix, feature = "test-utils"))]
fn serve_exits_successfully_on_sigint() {
    assert_server_exits_successfully_on_signal(libc::SIGINT);
}

#[test]
#[cfg(all(unix, feature = "test-utils"))]
fn serve_exits_successfully_on_sigterm() {
    assert_server_exits_successfully_on_signal(libc::SIGTERM);
}

#[test]
#[cfg(all(unix, feature = "test-utils"))]
fn serve_ignores_sigpipe() {
    let addr = reserve_loopback_addr();
    let mut server = spawn_protocol_test_server(&addr, None);
    wait_for_server(&mut server.child, &addr);

    send_signal(&server.child, libc::SIGPIPE);
    std::thread::sleep(Duration::from_millis(100));

    assert!(
        server.child.try_wait().expect("poll server").is_none(),
        "server should ignore SIGPIPE while serving"
    );

    send_signal(&server.child, libc::SIGINT);
    let status = wait_for_child_exit(&mut server.child);
    assert!(
        status.success(),
        "server should still exit cleanly after ignored SIGPIPE: {status}"
    );
}

#[cfg(all(unix, feature = "test-utils"))]
fn assert_server_exits_successfully_on_signal(signal: libc::c_int) {
    let addr = reserve_loopback_addr();
    let mut server = spawn_protocol_test_server(&addr, None);
    wait_for_server(&mut server.child, &addr);

    send_signal(&server.child, signal);
    let status = wait_for_child_exit(&mut server.child);

    assert!(
        status.success(),
        "server should exit cleanly after signal {signal}: {status}"
    );
}

#[cfg(all(unix, feature = "test-utils"))]
fn send_signal(child: &std::process::Child, signal: libc::c_int) {
    // SAFETY: `child.id()` belongs to the live server process managed by this test.
    let result = unsafe { libc::kill(child.id() as libc::pid_t, signal) };
    assert_eq!(
        result,
        0,
        "kill({signal}) failed: {}",
        std::io::Error::last_os_error()
    );
}

#[cfg(all(unix, feature = "test-utils"))]
fn wait_for_child_exit(child: &mut std::process::Child) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("poll server") {
            return status;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    child.kill().expect("kill hanging server");
    let _ = child.wait();
    panic!("server did not exit after shutdown signal");
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
