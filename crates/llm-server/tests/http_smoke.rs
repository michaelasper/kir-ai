#[cfg(feature = "test-utils")]
use std::{
    fmt,
    sync::{Arc, Mutex, OnceLock},
};
#[cfg(feature = "test-utils")]
use tracing::{
    Event, Id, Metadata, Subscriber,
    field::{Field, Visit},
    span,
};

#[cfg(feature = "test-utils")]
#[derive(Clone, Debug)]
struct RecordedEvent {
    fields: Vec<(String, String)>,
}

#[cfg(feature = "test-utils")]
impl RecordedEvent {
    fn field_contains(&self, name: &str, value: &str) -> bool {
        self.fields
            .iter()
            .any(|(field, recorded)| field == name && recorded.contains(value))
    }
}

#[cfg(feature = "test-utils")]
static TRACE_EVENTS: OnceLock<Arc<Mutex<Vec<RecordedEvent>>>> = OnceLock::new();

#[cfg(feature = "test-utils")]
struct TraceCapture {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

#[cfg(feature = "test-utils")]
impl TraceCapture {
    fn start() -> Self {
        let events = Arc::clone(TRACE_EVENTS.get_or_init(|| {
            let events = Arc::new(Mutex::new(Vec::new()));
            let subscriber = RecordingSubscriber {
                events: Arc::clone(&events),
            };
            tracing::subscriber::set_global_default(subscriber)
                .expect("trace test subscriber installs once");
            events
        }));
        events.lock().expect("recorded events lock").clear();
        tracing::callsite::rebuild_interest_cache();
        Self { events }
    }

    fn events(&self) -> Vec<RecordedEvent> {
        self.events.lock().expect("recorded events lock").clone()
    }
}

#[cfg(feature = "test-utils")]
struct RecordingSubscriber {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

#[cfg(feature = "test-utils")]
impl Subscriber for RecordingSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn register_callsite(
        &self,
        _metadata: &'static Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = FieldRecorder::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("recorded events lock")
            .push(RecordedEvent {
                fields: visitor.fields,
            });
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[cfg(feature = "test-utils")]
#[derive(Default)]
struct FieldRecorder {
    fields: Vec<(String, String)>,
}

#[cfg(feature = "test-utils")]
impl FieldRecorder {
    fn record_value(&mut self, field: &Field, value: String) {
        self.fields.push((field.name().to_owned(), value));
    }
}

#[cfg(feature = "test-utils")]
impl Visit for FieldRecorder {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn protocol_router_serves_chat_without_engine_crate() {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    let response = llm_server::build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_server::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat request reaches router");

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let body: Value = serde_json::from_slice(&bytes).expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "hello from rust native backend"
    );
}

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn protocol_router_emits_structured_access_logs_with_request_id() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    let trace = TraceCapture::start();
    let response = llm_server::build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("x-request-id", "access-log-health")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("health request reaches router");

    assert_eq!(response.status(), StatusCode::OK);
    let events = trace.events();
    assert!(
        events.iter().any(|event| {
            event.field_contains("message", "http request started")
                && event.field_contains("request_id", "access-log-health")
                && event.field_contains("method", "GET")
                && event.field_contains("path", "/health")
        }),
        "expected structured request-start access log, got {events:#?}"
    );
    assert!(
        events.iter().any(|event| {
            event.field_contains("message", "http request completed")
                && event.field_contains("request_id", "access-log-health")
                && event.field_contains("method", "GET")
                && event.field_contains("path", "/health")
                && event.field_contains("status", "200")
                && event.fields.iter().any(|(field, _)| field == "latency_ms")
        }),
        "expected structured request-completion access log, got {events:#?}"
    );
}

#[tokio::test]
async fn tls_serve_rejects_missing_certificate_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let err = llm_server::serve_tls(
        listener,
        axum::Router::new(),
        llm_server::TlsConfig::new(
            temp.path().join("missing-cert.pem"),
            temp.path().join("key.pem"),
        ),
    )
    .await
    .expect_err("missing certificate path fails before serving");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("TLS certificate file"),
        "error should identify the certificate path class: {err}"
    );
}

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn tls_serve_answers_https_health_on_loopback() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cert_path = temp.path().join("localhost-cert.pem");
    let key_path = temp.path().join("localhost-key.pem");
    tokio::fs::write(&cert_path, TEST_LOCALHOST_CERT)
        .await
        .expect("write cert");
    tokio::fs::write(&key_path, TEST_LOCALHOST_KEY)
        .await
        .expect("write key");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("local addr");
    let router = llm_server::build_router_with_protocol_test_backend();
    let server = tokio::spawn(llm_server::serve_tls(
        listener,
        router,
        llm_server::TlsConfig::new(&cert_path, &key_path),
    ));

    let client = reqwest::Client::builder()
        .no_proxy()
        .add_root_certificate(
            reqwest::Certificate::from_pem(TEST_LOCALHOST_CERT.as_bytes())
                .expect("test certificate parses"),
        )
        .build()
        .expect("HTTPS client builds");
    let response = client
        .get(format!("https://127.0.0.1:{}/health", addr.port()))
        .send()
        .await
        .expect("HTTPS request succeeds");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    server.abort();
}

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn tls_serve_idle_tcp_client_does_not_block_next_https_health_request() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cert_path = temp.path().join("localhost-cert.pem");
    let key_path = temp.path().join("localhost-key.pem");
    tokio::fs::write(&cert_path, TEST_LOCALHOST_CERT)
        .await
        .expect("write cert");
    tokio::fs::write(&key_path, TEST_LOCALHOST_KEY)
        .await
        .expect("write key");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("local addr");
    let router = llm_server::build_router_with_protocol_test_backend();
    let server = tokio::spawn(llm_server::serve_tls(
        listener,
        router,
        llm_server::TlsConfig::new(&cert_path, &key_path),
    ));

    let _idle_client = tokio::net::TcpStream::connect(addr)
        .await
        .expect("idle raw TCP client connects");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::builder()
        .no_proxy()
        .add_root_certificate(
            reqwest::Certificate::from_pem(TEST_LOCALHOST_CERT.as_bytes())
                .expect("test certificate parses"),
        )
        .build()
        .expect("HTTPS client builds");
    let response = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        client
            .get(format!("https://127.0.0.1:{}/health", addr.port()))
            .send(),
    )
    .await
    .expect("second HTTPS request is not blocked by idle TCP client")
    .expect("HTTPS request succeeds");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    server.abort();
}

const TEST_LOCALHOST_CERT: &str = r#"-----BEGIN CERTIFICATE-----
MIIDSTCCAjGgAwIBAgIUPio/cBD2WRZGNf5BEi5QsbkyGkAwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDUyMTA0NDUzOVoXDTM2MDUx
ODA0NDUzOVowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAkP2KiqVMUP3UCr2E/a6vIoBzKcuQk8Aydv/CrXNLNzB+
RsavV/HdY2Ttfrfj5je66wOJHZAkd83ohheh3rqCtWjBcCHJ8sU8V3CjYjfDzvfv
p7XumcmUtKPQZg6sQrCxXJUnnO1eBt3dWn5TvFT0F0OCfx5gn6l8bHGWkfyAT83A
cFuTE6TfzykJB49vhqBhV1m6Uo0wZzJ+/GTOnyM2ia9eUz9wx2jgcWNrM8hhcgXT
N/jPVuC+MGM0E4sIiRB19sAoJfUgmUqbxbVmpL0++VKrj4TB15uV+pcwfVqHl3Ba
FsY3uZL1tgOe/Ml9L5OBC/La5BenrXdhAE8bhp17UwIDAQABo4GSMIGPMB0GA1Ud
DgQWBBQDFIXbkufzBdLhwW8XIWuGFgRcnzAfBgNVHSMEGDAWgBQDFIXbkufzBdLh
wW8XIWuGFgRcnzAaBgNVHREEEzARgglsb2NhbGhvc3SHBH8AAAEwDAYDVR0TAQH/
BAIwADAOBgNVHQ8BAf8EBAMCBaAwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDQYJKoZI
hvcNAQELBQADggEBAHsfl7iZdJSIiSTXPPi/v0QJ19F12fjKbXgXrph/SQzYj5eB
eeYJkY94p2ItL3MmJ4dmLDkOqysAZ9Ogja8SZLh/1fYwNtr4en8u7immrg8nTaZ3
OG0Z3FBXbdEjI6kTqn6AmGdbcElX+vePHHIVY4obEGrMr2G4BDGjFuhaENYFsDd9
WbEKW2RycWTGw4Lk4ToVXymV2UZi39pu/Et+40tNdKheXh2ic6PpUNOoo5j+A9ug
9szDZdLA3zK5UUYPwzNiEVSiKW7B1o1atMvRmbKZo8BaBthLIAG5UFvuKHZPVHe2
7QhjXxdRRSMEhk/moXJYvTDVD44fo3+6/Zu84HM=
-----END CERTIFICATE-----
"#;

const TEST_LOCALHOST_KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCQ/YqKpUxQ/dQK
vYT9rq8igHMpy5CTwDJ2/8Ktc0s3MH5Gxq9X8d1jZO1+t+PmN7rrA4kdkCR3zeiG
F6HeuoK1aMFwIcnyxTxXcKNiN8PO9++nte6ZyZS0o9BmDqxCsLFclSec7V4G3d1a
flO8VPQXQ4J/HmCfqXxscZaR/IBPzcBwW5MTpN/PKQkHj2+GoGFXWbpSjTBnMn78
ZM6fIzaJr15TP3DHaOBxY2szyGFyBdM3+M9W4L4wYzQTiwiJEHX2wCgl9SCZSpvF
tWakvT75UquPhMHXm5X6lzB9WoeXcFoWxje5kvW2A578yX0vk4EL8trkF6etd2EA
TxuGnXtTAgMBAAECggEABTQFtRMLvMfwxGmBUv9Ii3+5++QXx9pr+5qssfMDKubD
gvO3Y2JFJ4xKlgBKBL8d2jg+wyZWM+YNgN5q2AAWfiWIjgC20bUKHmzNS2vxfxIL
vv3uIziGOYzvBvKVp2Q8gTRcskRPpwYIRXdm6hY1/XqJPaZph+FNTn3HDjpWOnoV
P/mSbzSysI/ul6W+Tqwy/TDblZvVIwUcswhZhuNwV8SbexF6FQ+JETvLcCkrmAJN
vZGxDULorDH2YFSXW+JrjKdITBqp4li5UbaD6Dn2agAfTVSaW4vPoTVy1c3PzZvH
okIAvHXJTycxD0x0rAZBvruJxw1c0RwmoHuDDbSP3QKBgQDHohVkJ2PZcDAx+q7m
I2sP3xyxS8fkltiPUBVhyHMMVQ1rv/V3k9OmTIJkA661k8WcemC/QMV5ItNoHh91
6KUAWJKuLezIy26ftc2LpX72nHEo0p1jglbhgqG+CIXQJQObGbP3VOe4BmBmuJD6
mBVNhVgJTiPS7JV6JI1qWqB93wKBgQC57cL0doJpFC4auTOdUN3c57+OyyngtmK9
bIUXeco7CGQo+CyOopLOwgFnEPJvwOqhy7VGgTIUowoA75Qeryj7cE45LEt7V4pc
9GrzM0lyMnsHpYtfDpi/zvPdL+n+4b7jn2zbfrmGcmCzhvb50Ljid7zoqaC6lt7c
ZPz3hL3JDQKBgQCg9zz+S5CEI6SIuBPMNuS9oG23O15LH6IwNCd5d7HkULQInHgl
Wcm/flNop1t4x1UALeDSdTyExyLlAdzmKpbYp5Jl5VvWL8nb9zBsGB4+ZLgNbX1A
XjkFjloyKxcSVLYKmnf0xr4sMOAME2e612Pd5NWucxYJnX+NQ+nOxpI/ywKBgQCl
LczIgEyFa+81wJlRRpmEesLc6jNPNtlr7fAjlgiK/350Q17abSY912+FkDHCBMKu
cRqgA4FpghsOD8oopHalQvXLp0V7057RzDcDzumOMbjJZ1H1ZjNgHEzckYex7/41
nNoJ+oB6KD0u4VWjRMIsODI1BRYNDqH5bSKsB1rQNQKBgDCjpj2MBMyjeqGBTGL8
syr0YY7HuCGf2by4DhOoPhqBnBMf1ngxQULYnZ9lK6dbbdBc/eUXp73lEkvF6yL9
OwC2zz+2g8kikO5NXycwI64Qc5A1KkSfsoDq+GcIE0tPQdwPXI1tMs/V9HYRjqMa
K5hxwT0jNtyM9+tiImDhNY9y
-----END PRIVATE KEY-----
"#;
