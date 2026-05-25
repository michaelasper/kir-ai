use super::*;
use axum::http::HeaderValue;

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

fn completion_request(prompt: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "prompt": prompt
            })
            .to_string(),
        ))
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
async fn public_inference_rate_limit_is_per_forwarded_client_for_chat_and_completion_routes() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_header(
            chat_request_body("allowed"),
            "x-forwarded-for",
            "203.0.113.10",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);

    let second = app
        .clone()
        .oneshot(with_header(
            completion_request("other client remains allowed"),
            "x-forwarded-for",
            "203.0.113.20",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::OK);

    let third = app
        .oneshot(with_header(
            completion_request("first client is limited"),
            "x-forwarded-for",
            "203.0.113.10",
        ))
        .await
        .expect("third response");
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
}

#[tokio::test]
async fn public_inference_rate_limit_is_per_real_ip_when_forwarded_for_is_absent() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(with_header(
            malformed_chat_request(),
            "x-real-ip",
            "198.51.100.10",
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

    let second = app
        .clone()
        .oneshot(with_header(
            malformed_chat_request(),
            "x-real-ip",
            "198.51.100.20",
        ))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::BAD_REQUEST);

    let third = app
        .oneshot(with_header(
            malformed_chat_request(),
            "x-real-ip",
            "198.51.100.10",
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
