use super::*;
use llm_server::{ServerBackendMetrics, ServerBackendMetricsSnapshot};

#[derive(Debug)]
struct PagedKvBackendMetrics;

impl ServerBackendMetrics for PagedKvBackendMetrics {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot {
        ServerBackendMetricsSnapshot {
            metrics: [(
                "paged_kv_cache".to_owned(),
                json!({
                    "resident_blocks": 2,
                    "active_blocks": 3,
                    "shared_blocks": 1,
                    "total_cow_clones": 1,
                    "blocks_evicted_lru": 2,
                    "pool_utilization_pct": 66.66666666666666,
                }),
            )]
            .into_iter()
            .collect(),
        }
    }

    fn kv_cache_snapshot(&self) -> Option<Value> {
        Some(json!({
            "object": "kv_cache.block_pool",
            "metrics": {
                "total_blocks": 3,
                "resident_blocks": 2,
                "active_blocks": 3,
                "free_list_blocks": 1,
                "shared_blocks": 1,
                "refcount_total": 3,
                "max_refcount_seen": 2,
                "total_cow_clones": 1,
                "cow_bytes_saved": 32,
                "blocks_evicted_lru": 2,
                "pool_utilization_pct": 66.66666666666666
            },
            "sessions": [
                {
                    "session_id": 7,
                    "layers": [
                        {
                            "layer": 0,
                            "block_table": [
                                {"index": 0, "block_id": 42, "ref_count": 2}
                            ]
                        }
                    ]
                }
            ],
            "blocks": [
                {"block_id": 42, "ref_count": 2, "token_count": 2}
            ]
        }))
    }
}

fn build_router_with_paged_kv_metrics() -> Router {
    router_builder(Box::new(StaticBackend {
        text: "unused".to_owned(),
    }))
    .with_metrics(Arc::new(PagedKvBackendMetrics))
    .allow_unauthenticated_admin()
    .build()
    .expect("test router builds")
}

fn assert_metric_incremented(before: &Value, after: &Value, path: &[&str], expected_delta: u64) {
    let before = metric_at_path(before, path);
    let after = metric_at_path(after, path);
    assert!(
        after >= before + expected_delta,
        "metric {path:?} should increase by at least {expected_delta}: before={before}, after={after}"
    );
}

fn assert_metric_unchanged(before: &Value, after: &Value, path: &[&str]) {
    let before = metric_at_path(before, path);
    let after = metric_at_path(after, path);
    assert_eq!(
        before, after,
        "metric {path:?} should not change: before={before}, after={after}"
    );
}

fn metric_at_path(metrics: &Value, path: &[&str]) -> u64 {
    let mut value = metrics;
    for segment in path {
        value = &value[*segment];
    }
    value.as_u64().expect("metric is an integer")
}

struct FakeMlxServer {
    endpoint: url::Url,
    snapshot: tempfile::TempDir,
    join: Option<thread::JoinHandle<()>>,
}

impl FakeMlxServer {
    fn start(response_body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
        let endpoint = url::Url::parse(&format!(
            "http://{}/v1",
            listener.local_addr().expect("local addr")
        ))
        .expect("endpoint url");
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fake mlx request");
            read_http_request(&mut stream);
            write_http_response(&mut stream, "200 OK", response_body);
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            join: Some(join),
        }
    }

    fn endpoint(&self) -> url::Url {
        self.endpoint.clone()
    }

    fn snapshot_path(&self) -> &Path {
        self.snapshot.path()
    }
}

impl Drop for FakeMlxServer {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            join.join().expect("fake MLX server exits");
        }
    }
}

fn read_http_request(stream: &mut std::net::TcpStream) {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end;
    loop {
        let read = stream.read(&mut buffer).expect("read request");
        assert!(read > 0, "client closed before headers");
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
    }
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("content length"))
        })
        .expect("content-length header");
    while bytes.len() < header_end + content_length {
        let read = stream.read(&mut buffer).expect("read body");
        assert!(read > 0, "client closed before body");
        bytes.extend_from_slice(&buffer[..read]);
    }
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}
