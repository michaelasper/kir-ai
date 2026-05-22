use llm_hub::{HubClient, HubFile, HubRepoId, HubTimeouts, ModelProfile, ModelStore};
use std::{
    io::{ErrorKind, Read, Write},
    net::TcpListener,
    thread,
    time::{Duration, Instant},
};
use url::Url;

#[test]
fn hub_client_new_returns_result_without_panicking() {
    let endpoint = Url::parse("https://huggingface.co").expect("test endpoint");

    let client = HubClient::new(endpoint).expect("hub client builds");

    drop(client);
}

#[test]
fn hub_client_with_timeouts_returns_result_without_panicking() {
    let endpoint = Url::parse("http://127.0.0.1:9").expect("test endpoint");

    let client = HubClient::with_timeouts(endpoint, short_timeouts()).expect("hub client builds");

    drop(client);
}

#[tokio::test]
async fn model_info_returns_network_error_when_request_times_out() {
    let (endpoint, server) = spawn_stalling_http_server(|_stream| {
        thread::sleep(Duration::from_millis(45));
    });
    let client = HubClient::with_timeouts(endpoint, short_timeouts()).expect("hub client builds");

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
async fn model_info_encodes_revision_as_path_segment() {
    let (endpoint, server) = spawn_stalling_http_server(|mut stream| {
        let mut buffer = [0_u8; 2048];
        let read = stream.read(&mut buffer).expect("read request");
        let request = String::from_utf8_lossy(&buffer[..read]);
        if request.starts_with(
            "GET /api/models/Qwen/Qwen3.6-35B-A3B/revision/refs%2Fpr%2F1?blobs=true&securityStatus=true ",
        ) {
            write_http_response(
                &mut stream,
                "200 OK",
                r#"{"id":"Qwen/Qwen3.6-35B-A3B","sha":"0123456789abcdef0123456789abcdef01234567","siblings":[]}"#,
            );
        } else {
            write_http_response(&mut stream, "400 Bad Request", r#"{"error":"bad path"}"#);
        }
    });
    let client = HubClient::with_timeouts(endpoint, short_timeouts()).expect("hub client builds");

    let info = client
        .model_info(
            &HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
            "refs/pr/1",
            None,
        )
        .await
        .expect("revision slash is encoded");

    assert_eq!(
        info.resolved_commit,
        "0123456789abcdef0123456789abcdef01234567"
    );
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
        thread::sleep(Duration::from_millis(45));
    });
    let client = HubClient::with_timeouts(
        endpoint,
        HubTimeouts {
            request: Duration::from_secs(10),
            ..short_timeouts()
        },
    )
    .expect("hub client builds");
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
    let staging_root = temp
        .path()
        .join("huggingface")
        .join("models--Qwen--Qwen3.6-35B-A3B")
        .join("staging");
    if tokio::fs::try_exists(&staging_root)
        .await
        .expect("check staging root")
    {
        let mut entries = tokio::fs::read_dir(&staging_root)
            .await
            .expect("read staging root");
        assert!(
            entries
                .next_entry()
                .await
                .expect("read staging entry")
                .is_none(),
            "failed pull should remove unique staging directories"
        );
    }
    server.join().expect("server exits");
}

fn short_timeouts() -> HubTimeouts {
    HubTimeouts {
        connect: Duration::from_millis(100),
        request: Duration::from_millis(40),
        read: Duration::from_millis(40),
    }
}

fn spawn_stalling_http_server(
    handler: impl FnOnce(std::net::TcpStream) + Send + 'static,
) -> (Url, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    listener
        .set_nonblocking(true)
        .expect("set fake server listener nonblocking");
    let endpoint = Url::parse(&format!(
        "http://{}",
        listener.local_addr().expect("local addr")
    ))
    .expect("endpoint URL");
    let server = thread::spawn(move || {
        let accept_deadline = Instant::now() + Duration::from_millis(250);
        let (stream, _) = loop {
            match listener.accept() {
                Ok(accepted) => break accepted,
                Err(err)
                    if err.kind() == ErrorKind::WouldBlock && Instant::now() < accept_deadline =>
                {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(err) => panic!("accept request: {err}"),
            }
        };
        stream
            .set_nonblocking(false)
            .expect("set fake server stream blocking");
        stream
            .set_read_timeout(Some(Duration::from_millis(250)))
            .expect("set fake server stream read timeout");
        handler(stream);
    });
    (endpoint, server)
}

fn write_http_response(stream: &mut std::net::TcpStream, status: &str, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write response");
    stream.flush().expect("flush response");
}
