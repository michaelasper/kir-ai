use llm_hub::{HubClient, HubFile, HubRepoId, HubTimeouts, ModelProfile, ModelStore};
use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
    time::{Duration, Instant},
};
use url::Url;

#[tokio::test]
async fn model_info_returns_network_error_when_request_times_out() {
    let (endpoint, server) = spawn_stalling_http_server(|mut stream| {
        let mut buffer = [0_u8; 1024];
        let _ = stream.read(&mut buffer);
        thread::sleep(Duration::from_millis(500));
    });
    let client = HubClient::with_timeouts(endpoint, short_timeouts());

    let started = Instant::now();
    let err = client
        .model_info(
            &HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
            "main",
            None,
        )
        .await
        .expect_err("stalled model info request times out");

    assert_eq!(err.code(), "model_download_interrupted");
    assert!(started.elapsed() < Duration::from_secs(2));
    server.join().expect("server exits");
}

#[tokio::test]
async fn pull_plan_returns_network_error_when_download_body_stalls() {
    let (endpoint, server) = spawn_stalling_http_server(|mut stream| {
        let mut buffer = [0_u8; 1024];
        let _ = stream.read(&mut buffer);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{")
            .expect("partial response");
        stream.flush().expect("flush response");
        thread::sleep(Duration::from_millis(500));
    });
    let client = HubClient::with_timeouts(
        endpoint,
        HubTimeouts {
            request: Duration::from_secs(10),
            ..short_timeouts()
        },
    );
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = llm_hub::build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, None)],
        &[],
    )
    .expect("plan builds");

    let started = Instant::now();
    let err = store
        .pull_plan(&client, &plan, None)
        .await
        .expect_err("stalled download body times out");

    assert_eq!(err.code(), "model_download_interrupted");
    assert!(err.to_string().contains("stalled"));
    assert!(started.elapsed() < Duration::from_secs(2));
    server.join().expect("server exits");
}

fn short_timeouts() -> HubTimeouts {
    HubTimeouts {
        connect: Duration::from_millis(100),
        request: Duration::from_millis(100),
        read: Duration::from_millis(100),
    }
}

fn spawn_stalling_http_server(
    handler: impl FnOnce(std::net::TcpStream) + Send + 'static,
) -> (Url, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let endpoint = Url::parse(&format!(
        "http://{}",
        listener.local_addr().expect("local addr")
    ))
    .expect("endpoint URL");
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept request");
        handler(stream);
    });
    (endpoint, server)
}
