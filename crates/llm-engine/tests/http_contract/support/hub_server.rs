fn spawn_fake_hub_server(requests: usize) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake hub");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
    let server = thread::spawn(move || {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().expect("accept fake hub request");
            let mut buffer = [0_u8; 4096];
            let read = stream.read(&mut buffer).expect("read fake hub request");
            let request = String::from_utf8_lossy(&buffer[..read]);
            if request.starts_with(
                "GET /api/models/Qwen/Qwen3.6-35B-A3B/revision/main?blobs=true&securityStatus=true ",
            ) {
                let body = json!({
                    "id": "Qwen/Qwen3.6-35B-A3B",
                    "sha": "0123456789abcdef0123456789abcdef01234567",
                    "siblings": [
                        {"rfilename": "config.json", "size": 2, "blobId": "\"cfg\""},
                        {"rfilename": "model.safetensors", "size": 4, "blobId": "\"weights\""}
                    ]
                })
                .to_string();
                write_http_response(&mut stream, "200 OK", &body);
            } else if request.starts_with(
                "GET /Qwen/Qwen3.6-35B-A3B/resolve/0123456789abcdef0123456789abcdef01234567/config.json ",
            ) {
                write_http_response(&mut stream, "200 OK", "{}");
            } else {
                write_http_response(&mut stream, "404 Not Found", "not found");
            }
        }
    });
    (endpoint, server)
}

struct BlockingFakeHubServer {
    endpoint: String,
    server: thread::JoinHandle<()>,
    download_started: mpsc::UnboundedReceiver<String>,
    release_download: Arc<(Mutex<usize>, Condvar)>,
    max_active_downloads: Arc<Mutex<usize>>,
}

impl BlockingFakeHubServer {
    fn release_one_download(&self) {
        let (release_count, release) = &*self.release_download;
        let mut release_count = release_count.lock().expect("release lock");
        *release_count += 1;
        release.notify_one();
    }

    fn max_active_downloads(&self) -> usize {
        *self
            .max_active_downloads
            .lock()
            .expect("active download lock")
    }
}

fn spawn_blocking_fake_hub_server() -> BlockingFakeHubServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake hub");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
    let (download_started, download_started_rx) = mpsc::unbounded_channel();
    let release_download = Arc::new((Mutex::new(0_usize), Condvar::new()));
    let active_downloads = Arc::new(Mutex::new(0_usize));
    let max_active_downloads = Arc::new(Mutex::new(0_usize));
    let server_release_download = release_download.clone();
    let server_active_downloads = active_downloads.clone();
    let server_max_active_downloads = max_active_downloads.clone();
    let server = thread::spawn(move || {
        let mut handlers = Vec::new();
        for _ in 0..4 {
            let (stream, _) = listener.accept().expect("accept fake hub request");
            let handler_download_started = download_started.clone();
            let handler_release_download = server_release_download.clone();
            let handler_active_downloads = server_active_downloads.clone();
            let handler_max_active_downloads = server_max_active_downloads.clone();
            handlers.push(thread::spawn(move || {
                handle_blocking_fake_hub_request(
                    stream,
                    handler_download_started,
                    handler_release_download,
                    handler_active_downloads,
                    handler_max_active_downloads,
                );
            }));
        }
        drop(download_started);
        for handler in handlers {
            handler.join().expect("fake hub handler exits");
        }
    });

    BlockingFakeHubServer {
        endpoint,
        server,
        download_started: download_started_rx,
        release_download,
        max_active_downloads,
    }
}

fn handle_blocking_fake_hub_request(
    mut stream: std::net::TcpStream,
    download_started: mpsc::UnboundedSender<String>,
    release_download: Arc<(Mutex<usize>, Condvar)>,
    active_downloads: Arc<Mutex<usize>>,
    max_active_downloads: Arc<Mutex<usize>>,
) {
    let mut buffer = [0_u8; 4096];
    let read = stream.read(&mut buffer).expect("read fake hub request");
    let request = String::from_utf8_lossy(&buffer[..read]);
    let Some(repo_id) = blocking_fake_hub_repo_id(&request) else {
        write_http_response(&mut stream, "404 Not Found", "not found");
        return;
    };
    if request.starts_with(&format!(
        "GET /api/models/{repo_id}/revision/main?blobs=true&securityStatus=true "
    )) {
        let body = json!({
            "id": repo_id,
            "sha": "0123456789abcdef0123456789abcdef01234567",
            "siblings": [
                {"rfilename": "config.json", "size": 2, "blobId": "\"cfg\""},
                {"rfilename": "model.safetensors", "size": 4, "blobId": "\"weights\""}
            ]
        })
        .to_string();
        write_http_response(&mut stream, "200 OK", &body);
        return;
    }
    if request.starts_with(&format!(
        "GET /{repo_id}/resolve/0123456789abcdef0123456789abcdef01234567/config.json "
    )) {
        record_blocking_fake_hub_download_start(
            &repo_id,
            &download_started,
            &active_downloads,
            &max_active_downloads,
        );
        wait_for_blocking_fake_hub_release(&release_download);
        write_http_response(&mut stream, "200 OK", "{}");
        let mut active_downloads = active_downloads.lock().expect("active download lock");
        *active_downloads -= 1;
        return;
    }
    write_http_response(&mut stream, "404 Not Found", "not found");
}

fn blocking_fake_hub_repo_id(request: &str) -> Option<String> {
    for repo_id in ["TestOrg/FirstModel", "TestOrg/SecondModel"] {
        if request.contains(repo_id) {
            return Some(repo_id.to_owned());
        }
    }
    None
}

fn record_blocking_fake_hub_download_start(
    repo_id: &str,
    download_started: &mpsc::UnboundedSender<String>,
    active_downloads: &Arc<Mutex<usize>>,
    max_active_downloads: &Arc<Mutex<usize>>,
) {
    let active_download_count = {
        let mut active_downloads = active_downloads.lock().expect("active download lock");
        *active_downloads += 1;
        *active_downloads
    };
    let mut max_active_downloads = max_active_downloads.lock().expect("active download lock");
    *max_active_downloads = (*max_active_downloads).max(active_download_count);
    download_started
        .send(repo_id.to_owned())
        .expect("download started receiver is open");
}

fn wait_for_blocking_fake_hub_release(release_download: &Arc<(Mutex<usize>, Condvar)>) {
    let (release_count, release) = &**release_download;
    let mut release_count = release_count.lock().expect("release lock");
    while *release_count == 0 {
        release_count = release.wait(release_count).expect("release lock");
    }
    *release_count -= 1;
}

fn write_http_response(stream: &mut std::net::TcpStream, status: &str, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write fake hub response");
    stream.flush().expect("flush fake hub response");
}
