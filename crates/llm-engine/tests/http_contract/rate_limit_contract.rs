use super::*;
use axum::{extract::ConnectInfo, http::HeaderValue};
use std::net::SocketAddr;

fn router_with_public_inference_rate_limit(max_requests: usize) -> Router {
    build_router_with_backend_and_options_allowing_unauthenticated_admin(
        Box::new(StaticBackend {
            text: "rate limit response".to_owned(),
        }),
        EngineOptions {
            public_inference_rate_limit: PublicInferenceRateLimit {
                max_requests,
                window: Duration::from_secs(60),
            },
            ..EngineOptions::default()
        },
    )
    .expect("rate-limited router builds")
}

fn malformed_chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{not-json"))
        .expect("request builds")
}

fn with_header(
    mut request: Request<Body>,
    name: &'static str,
    value: &'static str,
) -> Request<Body> {
    request
        .headers_mut()
        .insert(name, HeaderValue::from_static(value));
    request
}

fn with_peer_addr(mut request: Request<Body>, addr: &'static str) -> Request<Body> {
    let addr: SocketAddr = addr.parse().expect("peer address parses");
    request.extensions_mut().insert(ConnectInfo(addr));
    request
}

#[tokio::test]
async fn public_inference_rate_limit_rejects_fast_repeated_invalid_chat_requests() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(malformed_chat_request())
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        first
            .headers()
            .get("x-ratelimit-limit-requests")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    assert_eq!(
        first
            .headers()
            .get("x-ratelimit-remaining-requests")
            .and_then(|value| value.to_str().ok()),
        Some("0")
    );
    assert_eq!(
        first
            .headers()
            .get("x-ratelimit-reset-requests")
            .and_then(|value| value.to_str().ok()),
        Some("60")
    );

    let second = app
        .oneshot(malformed_chat_request())
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        second
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("60")
    );
    assert_eq!(
        second
            .headers()
            .get("x-ratelimit-limit-requests")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    assert_eq!(
        second
            .headers()
            .get("x-ratelimit-remaining-requests")
            .and_then(|value| value.to_str().ok()),
        Some("0")
    );
    assert_eq!(
        second
            .headers()
            .get("x-ratelimit-reset-requests")
            .and_then(|value| value.to_str().ok()),
        Some("60")
    );
    let body = body_json(second.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
    assert_eq!(body["error"]["phase"], "rate_limit");
    assert_eq!(body["error"]["retryable"], true);
}

#[tokio::test]
async fn public_inference_rate_limit_uses_authorization_before_spoofable_forwarding_headers() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_header(
            with_header(
                malformed_chat_request(),
                "authorization",
                "Bearer stable-client",
            ),
            "x-forwarded-for",
            "203.0.113.10",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

    let second = app
        .clone()
        .oneshot(with_header(
            with_header(
                malformed_chat_request(),
                "authorization",
                "Bearer stable-client",
            ),
            "x-forwarded-for",
            "203.0.113.20",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(second.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");

    let third = app
        .oneshot(with_header(
            with_header(
                malformed_chat_request(),
                "authorization",
                "Bearer other-client",
            ),
            "x-forwarded-for",
            "203.0.113.10",
        ))
        .await
        .expect("third response");
    assert_eq!(third.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn public_inference_rate_limit_uses_socket_peer_and_ignores_spoofed_forwarding_headers() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_peer_addr(
            with_header(malformed_chat_request(), "x-forwarded-for", "203.0.113.10"),
            "198.51.100.10:5000",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

    let second = app
        .clone()
        .oneshot(with_peer_addr(
            with_header(malformed_chat_request(), "x-forwarded-for", "203.0.113.10"),
            "198.51.100.20:5000",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::BAD_REQUEST);

    let third = app
        .oneshot(with_peer_addr(
            with_header(malformed_chat_request(), "x-forwarded-for", "203.0.113.99"),
            "198.51.100.10:5001",
        ))
        .await
        .expect("third response");
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
}

#[tokio::test]
async fn public_inference_rate_limit_is_per_socket_peer_when_forwarding_headers_are_absent() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_peer_addr(
            malformed_chat_request(),
            "198.51.100.10:5000",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

    let second = app
        .clone()
        .oneshot(with_peer_addr(
            malformed_chat_request(),
            "198.51.100.20:5000",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::BAD_REQUEST);

    let third = app
        .oneshot(with_peer_addr(
            malformed_chat_request(),
            "198.51.100.10:5001",
        ))
        .await
        .expect("third response");
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
}

#[tokio::test]
async fn public_inference_rate_limit_is_per_authorization_header_when_ip_headers_are_absent() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_header(
            malformed_chat_request(),
            "authorization",
            "Bearer client-a",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

    let second = app
        .clone()
        .oneshot(with_header(
            malformed_chat_request(),
            "authorization",
            "Bearer client-b",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::BAD_REQUEST);

    let third = app
        .oneshot(with_header(
            malformed_chat_request(),
            "authorization",
            "Bearer client-a",
        ))
        .await
        .expect("third response");
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
}

#[tokio::test]
async fn public_inference_rate_limit_does_not_throttle_health_or_model_list() {
    let app = router_with_public_inference_rate_limit(1);

    let limited = app
        .clone()
        .oneshot(chat_request_body("consume public inference budget"))
        .await
        .expect("limited response");
    assert_eq!(limited.status(), StatusCode::OK);

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("health response");
    assert_eq!(health.status(), StatusCode::OK);

    let models = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("models response");
    assert_eq!(models.status(), StatusCode::OK);
}
