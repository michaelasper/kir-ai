use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::Value;
#[cfg(feature = "bench")]
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
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

async fn write_verified_metadata_only_snapshot(root: &Path) -> std::path::PathBuf {
    let store = ModelStore::new(root);
    let full_plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let metadata_plan = full_plan.metadata_only();
    let snapshot_path = store.snapshot_path(&metadata_plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    store
        .verify_existing_snapshot(&metadata_plan)
        .await
        .expect("snapshot verifies");
    snapshot_path
}

fn spawn_test_hub_server(
    handler: impl FnOnce(TcpListener) + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test hub endpoint");
    listener
        .set_nonblocking(true)
        .expect("test hub listener nonblocking");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
    let server = thread::spawn(move || handler(listener));
    (endpoint, server)
}

fn accept_hub_request(listener: &TcpListener) -> (String, TcpStream) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                stream
                    .set_nonblocking(false)
                    .expect("test hub stream blocking");
                let mut buffer = [0_u8; 4096];
                let read = stream.read(&mut buffer).expect("read hub request");
                return (
                    String::from_utf8_lossy(&buffer[..read]).into_owned(),
                    stream,
                );
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "test hub endpoint did not receive a request"
                );
                thread::sleep(Duration::from_millis(10));
            }
            Err(err) => panic!("accept hub request failed: {err}"),
        }
    }
}

#[cfg(unix)]
#[test]
fn accept_hub_request_returns_blocking_stream_from_nonblocking_listener() {
    use std::os::fd::AsRawFd;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
    listener
        .set_nonblocking(true)
        .expect("test listener nonblocking");
    let mut client = TcpStream::connect(listener.local_addr().expect("local addr"))
        .expect("connect to test listener");
    client
        .write_all(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
        .expect("write request");

    let (request, stream) = accept_hub_request(&listener);

    assert!(request.starts_with("GET / HTTP/1.1"), "request: {request}");
    // SAFETY: `stream.as_raw_fd()` is a valid descriptor for the live TcpStream.
    let flags = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
    assert_ne!(
        flags,
        -1,
        "fcntl F_GETFL failed: {}",
        std::io::Error::last_os_error()
    );
    assert_eq!(
        flags & libc::O_NONBLOCK,
        0,
        "accepted hub stream should be blocking"
    );
}

fn write_json_response(stream: &mut TcpStream, body: &str) {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .expect("write JSON response");
    stream.flush().expect("flush JSON response");
}

fn write_bytes_response(stream: &mut TcpStream, body: &[u8]) {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .expect("write bytes response headers");
    stream.write_all(body).expect("write bytes response body");
    stream.flush().expect("flush bytes response");
}

fn assert_model_info_request(request: &str) {
    assert!(
        request.starts_with("GET /api/models/Qwen/Qwen3.6-35B-A3B/revision/main?"),
        "request: {request}"
    );
    assert!(request.contains("blobs=true"), "request: {request}");
    assert!(
        request.contains("securityStatus=true"),
        "request: {request}"
    );
}

fn hub_model_info_body() -> String {
    serde_json::json!({
        "id": "Qwen/Qwen3.6-35B-A3B",
        "sha": "0123456789abcdef0123456789abcdef01234567",
        "siblings": [
            {"rfilename": "config.json", "size": 2}
        ]
    })
    .to_string()
}

#[cfg(feature = "test-utils")]
struct ServerProcess {
    child: std::process::Child,
}

#[cfg(feature = "test-utils")]
impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(feature = "test-utils")]
fn reserve_loopback_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
    listener
        .local_addr()
        .expect("reserved loopback addr")
        .to_string()
}

#[cfg(feature = "test-utils")]
fn spawn_protocol_test_server(addr: &str, admin_token: Option<&str>) -> ServerProcess {
    let mut command = Command::new(env!("CARGO_BIN_EXE_llm-engine"));
    command
        .args([
            "serve",
            "--addr",
            addr,
            "--protocol-test-backend",
            "--i-understand-this-is-not-real-inference",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(token) = admin_token {
        command.args(["--admin-token", token]);
    }

    ServerProcess {
        child: command.spawn().expect("spawn protocol test server"),
    }
}

#[cfg(feature = "test-utils")]
fn wait_for_server(child: &mut std::process::Child, addr: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("poll protocol test server") {
            panic!("protocol test server exited before accepting requests: {status}");
        }
        if http_get(addr, "/health", None)
            .map(|response| response.starts_with("HTTP/1.1 200"))
            .unwrap_or(false)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    panic!("protocol test server did not accept requests at {addr}");
}

#[cfg(feature = "test-utils")]
fn http_get(addr: &str, path: &str, extra_headers: Option<&str>) -> std::io::Result<String> {
    use std::io::{Read, Write};

    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let extra_headers = extra_headers.unwrap_or("");
    let request =
        format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\n{extra_headers}Connection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}
