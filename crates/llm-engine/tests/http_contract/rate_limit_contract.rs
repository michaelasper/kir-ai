use super::*;

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

#[tokio::test]
async fn public_inference_rate_limit_rejects_fast_repeated_invalid_chat_requests() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(malformed_chat_request())
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);

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
    let body = body_json(second.into_body()).await;
    assert_eq!(body["error"]["code"], "rate_limited");
    assert_eq!(body["error"]["phase"], "rate_limit");
    assert_eq!(body["error"]["retryable"], true);
}

#[tokio::test]
async fn public_inference_rate_limit_is_global_for_chat_and_completion_routes() {
    let app = router_with_public_inference_rate_limit(1);

    let first = app
        .clone()
        .oneshot(chat_request_body("allowed"))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);

    let second = app
        .oneshot(completion_request("limited"))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(second.into_body()).await;
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
