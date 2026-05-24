use super::*;

#[tokio::test]
async fn mlx_backend_rejects_model_mismatch_before_http_request() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let err = backend
        .generate(BackendRequest::raw_completion(
            "other-model",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("model mismatch fails before HTTP");

    assert!(err.is_model_not_found());
}

#[tokio::test]
async fn mlx_backend_rejects_non_loopback_endpoint() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("https://example.com/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("remote MLX endpoint is rejected");

    assert!(err.to_string().contains("not loopback"));
}

#[tokio::test]
async fn mlx_backend_rejects_manifestless_snapshot_without_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("raw MLX family is required");

    assert!(
        err.to_string()
            .contains("MLX backend requires model family metadata")
    );
}

#[tokio::test]
async fn mlx_backend_accepts_gemma_requested_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("Gemma MLX backend opens");

    assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
}

#[tokio::test]
async fn mlx_backend_rejects_non_mlx_manifest_loader() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    write_mlx_manifest(snapshot.path(), "native-metal", "qwen");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("MLX backend rejects native manifest loader");

    assert!(
        err.to_string()
            .contains("MLX backend requires manifest loader `mlx`")
    );
}

#[tokio::test]
async fn mlx_backend_accepts_llama_requested_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            family: Some(ModelFamily::Llama),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("Llama MLX backend opens");

    assert_eq!(backend.model_metadata().family.as_deref(), Some("llama"));
}

#[tokio::test]
async fn mlx_backend_rejects_unknown_manifest_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    write_mlx_manifest(snapshot.path(), "mlx", "glm");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("unknown manifest family is rejected");

    assert!(err.to_string().contains("unsupported model family `glm`"));
}

#[tokio::test]
async fn mlx_backend_rejects_sse_without_done_marker() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"partial\",\"finish_reason\":\"stop\"}]}\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("missing DONE fails closed");

    assert!(err.to_string().contains("[DONE]"));
}

#[tokio::test]
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_per_chunk_timeout_detects_stalled_stream() {
    let server = FakeMlxServer::start_with_stall(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\n",
        Duration::from_millis(80),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_secs(5),
                read: Duration::from_millis(40),
            },
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("stalled stream produces timeout error");

    assert!(
        err.to_string().contains("stalled"),
        "expected stall error, got: {err}"
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}

#[tokio::test]
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_read_timeout_allows_initial_prefill_silence() {
    let server = FakeMlxServer::start_with_initial_body_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
        Duration::from_millis(60),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_secs(5),
                read: Duration::from_millis(30),
            },
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let chunks = backend
        .generate_stream(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("initial prefill silence should not trip read timeout");

    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(text, "late");
    let metrics = metrics.snapshot();
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["stall_failures"], 0);
}

#[tokio::test]
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_request_timeout_detects_delayed_response_headers() {
    let server = FakeMlxServer::start_with_response_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        Duration::from_millis(60),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_millis(30),
                read: Duration::from_secs(5),
            },
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("delayed response headers produce timeout error");

    assert!(
        err.to_string().contains("timed out"),
        "expected timeout error, got: {err}"
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}

#[tokio::test]
async fn mlx_backend_health_reports_model_list_http_failure() {
    let server = FakeMlxServer::start_with_status(503, "Service Unavailable", "{}");
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let health = backend.health().await;

    assert_eq!(health.status().as_str(), "unavailable");
    assert_eq!(server.received_path(), "/v1/models");
    assert!(
        health
            .reason()
            .expect("unavailable health reports a reason")
            .contains("503"),
        "health reason should include upstream status: {health:?}"
    );
}
