use crate::manifest::verify_file_sha256_for_artifact;
use crate::plan::validate_artifact_path;
use crate::{
    ArtifactClass, DownloadPlan, HubError, HubModelInfo, HubRepoId, ModelProfile,
    build_download_plan,
};
use futures::StreamExt;
use serde_json::Value;
use std::{path::Path, time::Duration};
use tokio::io::AsyncWriteExt;
use tokio::time;
use url::Url;

fn set_hub_path<'a>(
    url: &mut Url,
    segments: impl IntoIterator<Item = &'a str>,
) -> Result<(), HubError> {
    let mut path_segments = url
        .path_segments_mut()
        .map_err(|_| HubError::invalid_request("Hub endpoint must be hierarchical"))?;
    path_segments.clear();
    for segment in segments {
        path_segments.push(segment);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct HubClient {
    endpoint: Url,
    client: reqwest::Client,
    timeouts: HubTimeouts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HubTimeouts {
    pub connect: Duration,
    pub request: Duration,
    pub read: Duration,
}

impl Default for HubTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(15),
            request: Duration::from_secs(60 * 60 * 6),
            read: Duration::from_secs(120),
        }
    }
}

const DEFAULT_HUB_ENDPOINT: &str = "https://huggingface.co";

impl HubClient {
    pub fn default_client() -> Result<Self, HubError> {
        let endpoint = Url::parse(DEFAULT_HUB_ENDPOINT).map_err(|err| {
            HubError::invalid_request(format!(
                "default hub endpoint `{DEFAULT_HUB_ENDPOINT}` is invalid: {err}"
            ))
        })?;
        Self::new(endpoint)
    }

    pub fn new(endpoint: Url) -> Result<Self, HubError> {
        Self::with_timeouts(endpoint, HubTimeouts::default())
    }

    pub fn with_timeouts(endpoint: Url, timeouts: HubTimeouts) -> Result<Self, HubError> {
        Ok(Self {
            endpoint,
            client: build_http_client(timeouts)?,
            timeouts,
        })
    }

    pub async fn model_info(
        &self,
        repo_id: &HubRepoId,
        revision: &str,
        token: Option<&str>,
    ) -> Result<HubModelInfo, HubError> {
        let mut url = self.endpoint.clone();
        let (namespace, name) = repo_id.components()?;
        set_hub_path(
            &mut url,
            ["api", "models", namespace, name, "revision", revision],
        )?;
        let mut request = self
            .client
            .get(url)
            .query(&[("blobs", "true"), ("securityStatus", "true")]);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.map_err(HubError::network)?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(HubError::auth_failed("Hugging Face authentication failed"));
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(HubError::model_not_found(format!(
                "model repo `{}` revision `{revision}` was not found",
                repo_id.as_str()
            )));
        }
        if !status.is_success() {
            return Err(HubError::network(format!(
                "Hugging Face API returned HTTP {status}"
            )));
        }
        let value = response.json::<Value>().await.map_err(HubError::network)?;
        HubModelInfo::from_api_json(value)
    }

    pub async fn plan_model(
        &self,
        repo_id: HubRepoId,
        revision: &str,
        profile: ModelProfile,
        token: Option<&str>,
    ) -> Result<DownloadPlan, HubError> {
        let info = self.model_info(&repo_id, revision, token).await?;
        build_download_plan(
            repo_id,
            revision,
            info.resolved_commit,
            profile,
            info.files,
            &[],
        )
    }

    pub(crate) async fn download_file_to(
        &self,
        request: HubDownloadFileRequest<'_>,
    ) -> Result<(), HubError> {
        let HubDownloadFileRequest {
            repo_id,
            resolved_commit,
            path,
            destination,
            expected_size,
            expected_sha256,
            artifact_class,
            token,
        } = request;
        validate_artifact_path(path)?;
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(HubError::io)?;
        }
        let existing_len = match tokio::fs::metadata(destination).await {
            Ok(metadata) if metadata.len() == expected_size => {
                verify_file_sha256_for_artifact(destination, expected_sha256, artifact_class)
                    .await?;
                return Ok(());
            }
            Ok(metadata) if metadata.len() < expected_size => metadata.len(),
            Ok(_) => {
                tokio::fs::remove_file(destination)
                    .await
                    .map_err(HubError::io)?;
                0
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
            Err(err) => return Err(HubError::io(err)),
        };
        let mut url = self.endpoint.clone();
        let (namespace, name) = repo_id.components()?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| HubError::invalid_request("Hub endpoint must be hierarchical"))?;
            segments
                .clear()
                .push(namespace)
                .push(name)
                .push("resolve")
                .push(resolved_commit);
            for component in path.split('/') {
                segments.push(component);
            }
        }
        let mut request = self.client.get(url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        if existing_len > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing_len}-"));
        }
        let response = request.send().await.map_err(HubError::network)?;
        let status = response.status();
        if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) {
            return Err(HubError::network(format!(
                "download for `{path}` returned HTTP {status}"
            )));
        }
        let append_partial = existing_len > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        let expected_response_len = expected_download_response_len(
            path,
            status,
            response.headers(),
            existing_len,
            expected_size,
        )?;
        let mut file = if existing_len > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(destination)
                .await
                .map_err(HubError::io)?
        } else {
            tokio::fs::File::create(destination)
                .await
                .map_err(HubError::io)?
        };
        let mut stream = response.bytes_stream();
        let mut written_len = 0_u64;
        while let Some(chunk) = time::timeout(self.timeouts.read, stream.next())
            .await
            .map_err(|_| {
                HubError::network(format!(
                    "download for `{path}` stalled for {} while reading response body",
                    format_duration(self.timeouts.read)
                ))
            })?
        {
            let chunk = chunk.map_err(HubError::network)?;
            written_len = written_len.checked_add(chunk.len() as u64).ok_or_else(|| {
                HubError::integrity_failed(format!(
                    "downloaded `{path}` response body length overflowed u64"
                ))
            })?;
            file.write_all(&chunk).await.map_err(HubError::io)?;
        }
        file.flush().await.map_err(HubError::io)?;
        if written_len != expected_response_len {
            let mode = if append_partial { "resumed" } else { "full" };
            return Err(HubError::integrity_failed(format!(
                "{mode} download for `{path}` wrote {written_len} bytes, expected {expected_response_len}"
            )));
        }
        let final_len = tokio::fs::metadata(destination)
            .await
            .map_err(HubError::io)?
            .len();
        if final_len != expected_size {
            return Err(HubError::integrity_failed(format!(
                "downloaded `{path}` size {final_len} did not match expected {expected_size}"
            )));
        }
        verify_file_sha256_for_artifact(destination, expected_sha256, artifact_class).await?;
        Ok(())
    }
}

fn expected_download_response_len(
    path: &str,
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    existing_len: u64,
    expected_size: u64,
) -> Result<u64, HubError> {
    if status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Ok(expected_size);
    }
    if existing_len == 0 {
        return Err(HubError::integrity_failed(format!(
            "download for `{path}` returned unexpected HTTP 206 Partial Content without a resume request"
        )));
    }
    validate_resume_content_range(path, headers, existing_len, expected_size)?;
    expected_size.checked_sub(existing_len).ok_or_else(|| {
        HubError::integrity_failed(format!(
            "resume offset {existing_len} exceeds expected size {expected_size} for `{path}`"
        ))
    })
}

fn validate_resume_content_range(
    path: &str,
    headers: &reqwest::header::HeaderMap,
    existing_len: u64,
    expected_size: u64,
) -> Result<(), HubError> {
    let value = headers
        .get(reqwest::header::CONTENT_RANGE)
        .ok_or_else(|| {
            HubError::integrity_failed(format!(
                "resumed download for `{path}` returned HTTP 206 without Content-Range"
            ))
        })?
        .to_str()
        .map_err(|err| {
            HubError::integrity_failed(format!(
                "resumed download for `{path}` returned invalid Content-Range header: {err}"
            ))
        })?;
    let (start, end, total) = parse_content_range(value).ok_or_else(|| {
        HubError::integrity_failed(format!(
            "resumed download for `{path}` returned unsupported Content-Range `{value}`"
        ))
    })?;
    let expected_end = expected_size.checked_sub(1).ok_or_else(|| {
        HubError::integrity_failed(format!(
            "resumed download for `{path}` has zero expected size"
        ))
    })?;
    if start != existing_len || end != expected_end || total != expected_size {
        return Err(HubError::integrity_failed(format!(
            "resumed download for `{path}` returned Content-Range `{value}`, expected bytes {existing_len}-{expected_end}/{expected_size}"
        )));
    }
    Ok(())
}

fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    let total = total.parse().ok()?;
    (start <= end).then_some((start, end, total))
}

fn build_http_client(timeouts: HubTimeouts) -> Result<reqwest::Client, HubError> {
    let result = reqwest::Client::builder()
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build();
    map_http_client_build_result(result)
}

fn map_http_client_build_result(
    result: Result<reqwest::Client, impl ToString>,
) -> Result<reqwest::Client, HubError> {
    result.map_err(|err| {
        let message = err.to_string();
        HubError::network(format!("failed to build hub HTTP client: {message}"))
    })
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

pub(crate) struct HubDownloadFileRequest<'a> {
    pub(crate) repo_id: &'a HubRepoId,
    pub(crate) resolved_commit: &'a str,
    pub(crate) path: &'a str,
    pub(crate) destination: &'a Path,
    pub(crate) expected_size: u64,
    pub(crate) expected_sha256: Option<&'a str>,
    pub(crate) artifact_class: ArtifactClass,
    pub(crate) token: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    #[test]
    fn http_client_build_failure_maps_to_hub_error() {
        let result: Result<reqwest::Client, &str> = Err("TLS backend unavailable");

        let err = map_http_client_build_result(result).expect_err("build failure maps to HubError");

        assert_eq!(err.code(), "model_download_interrupted");
        assert!(err.to_string().contains("failed to build hub HTTP client"));
        assert!(err.to_string().contains("TLS backend unavailable"));
    }

    #[tokio::test]
    async fn resumed_download_rejects_missing_content_range_even_when_size_matches() {
        let temp = tempfile::tempdir().expect("tempdir");
        let destination = temp.path().join("config.json");
        tokio::fs::write(&destination, "ab")
            .await
            .expect("partial file");
        let (endpoint, server) = spawn_test_hub_server(|mut stream| {
            let request = read_http_request(&mut stream);
            assert!(
                request.to_ascii_lowercase().contains("range: bytes=2-"),
                "request: {request}"
            );
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nConnection: close\r\n\r\ncdef"
            )
            .expect("write response");
            stream.flush().expect("flush response");
        });
        let client =
            HubClient::with_timeouts(endpoint, test_timeouts()).expect("hub client builds");
        let repo_id = test_repo_id();

        let err = client
            .download_file_to(test_download_request(&repo_id, &destination, 6))
            .await
            .expect_err("missing content-range fails closed");

        assert_eq!(err.code(), "model_integrity_failed");
        assert!(err.to_string().contains("Content-Range"), "err: {err}");
        server.join().expect("server exits");
    }

    #[tokio::test]
    async fn resumed_download_rejects_wrong_content_range_start_even_when_size_matches() {
        let temp = tempfile::tempdir().expect("tempdir");
        let destination = temp.path().join("config.json");
        tokio::fs::write(&destination, "ab")
            .await
            .expect("partial file");
        let (endpoint, server) = spawn_test_hub_server(|mut stream| {
            let request = read_http_request(&mut stream);
            assert!(
                request.to_ascii_lowercase().contains("range: bytes=2-"),
                "request: {request}"
            );
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/6\r\nContent-Length: 4\r\nConnection: close\r\n\r\ncdef"
            )
            .expect("write response");
            stream.flush().expect("flush response");
        });
        let client =
            HubClient::with_timeouts(endpoint, test_timeouts()).expect("hub client builds");
        let repo_id = test_repo_id();

        let err = client
            .download_file_to(test_download_request(&repo_id, &destination, 6))
            .await
            .expect_err("wrong content-range start fails closed");

        assert_eq!(err.code(), "model_integrity_failed");
        assert!(err.to_string().contains("Content-Range"), "err: {err}");
        server.join().expect("server exits");
    }

    #[tokio::test]
    async fn resumed_download_accepts_matching_content_range() {
        let temp = tempfile::tempdir().expect("tempdir");
        let destination = temp.path().join("config.json");
        tokio::fs::write(&destination, "ab")
            .await
            .expect("partial file");
        let (endpoint, server) = spawn_test_hub_server(|mut stream| {
            let request = read_http_request(&mut stream);
            assert!(
                request.to_ascii_lowercase().contains("range: bytes=2-"),
                "request: {request}"
            );
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 2-5/6\r\nContent-Length: 4\r\nConnection: close\r\n\r\ncdef"
            )
            .expect("write response");
            stream.flush().expect("flush response");
        });
        let client =
            HubClient::with_timeouts(endpoint, test_timeouts()).expect("hub client builds");
        let repo_id = test_repo_id();

        client
            .download_file_to(test_download_request(&repo_id, &destination, 6))
            .await
            .expect("matching content-range resumes");

        let bytes = tokio::fs::read(&destination)
            .await
            .expect("read destination");
        assert_eq!(bytes, b"abcdef");
        server.join().expect("server exits");
    }

    #[tokio::test]
    async fn existing_weight_download_without_sha256_rejects_size_match_skip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let destination = temp.path().join("model.safetensors");
        tokio::fs::write(&destination, b"data")
            .await
            .expect("existing weights");
        let client = HubClient::with_timeouts(
            Url::parse("http://127.0.0.1:9").expect("test endpoint"),
            test_timeouts(),
        )
        .expect("hub client builds");
        let repo_id = test_repo_id();

        let err = client
            .download_file_to(HubDownloadFileRequest {
                repo_id: &repo_id,
                resolved_commit: "0123456789abcdef0123456789abcdef01234567",
                path: "model.safetensors",
                destination: &destination,
                expected_size: 4,
                expected_sha256: None,
                artifact_class: ArtifactClass::Weights,
                token: None,
            })
            .await
            .expect_err("missing weight digest must fail before download skip");

        assert_eq!(err.code(), "model_integrity_failed");
        assert!(err.to_string().contains("missing sha256"), "err: {err}");
    }

    fn test_repo_id() -> HubRepoId {
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id")
    }

    fn test_download_request<'a>(
        repo_id: &'a HubRepoId,
        destination: &'a Path,
        expected_size: u64,
    ) -> HubDownloadFileRequest<'a> {
        HubDownloadFileRequest {
            repo_id,
            resolved_commit: "0123456789abcdef0123456789abcdef01234567",
            path: "config.json",
            destination,
            expected_size,
            expected_sha256: None,
            artifact_class: ArtifactClass::Config,
            token: None,
        }
    }

    fn test_timeouts() -> HubTimeouts {
        HubTimeouts {
            connect: Duration::from_millis(100),
            request: Duration::from_secs(2),
            read: Duration::from_secs(2),
        }
    }

    fn spawn_test_hub_server(
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

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer).expect("read request");
        String::from_utf8_lossy(&buffer[..read]).into_owned()
    }
}
