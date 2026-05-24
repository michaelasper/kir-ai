use super::*;

#[tokio::test]
async fn admin_metrics_requires_bearer_token_when_configured() {
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/metrics")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "admin_auth_required");
    assert_eq!(body["error"]["phase"], "admin_auth");
}

#[tokio::test]
async fn admin_metrics_accepts_configured_bearer_token() {
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/metrics")
            .header("authorization", "Bearer secret-admin-token")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_metrics_request_cache_records_buffered_observation() {
    let app = build_router_with_unauthenticated_admin(Box::new(CachedUsageBackend {
        prompt_tokens: 100,
        prompt_cached_tokens: Some(64),
        completion_tokens: 5,
    }));
    let request_id = "cache-buffered";
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-request-id", request_id)
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["model"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(observation["streamed"], false);
    assert_eq!(observation["prompt_tokens"], 100);
    assert_eq!(observation["cached_tokens"], 64);
    assert_eq!(observation["uncached_tokens"], 36);
    assert_eq!(observation["cache_status"], "partial");
    assert!(observation["latency_ms"].as_u64().is_some());
}

#[tokio::test]
async fn admin_metrics_request_cache_records_stable_prefix_identity_for_repeat_agent_turns() {
    let app = build_router_with_unauthenticated_admin(Box::new(CacheTransitionBackend {
        prompt_tokens: 100,
        warm_cached_tokens: 100,
        completion_tokens: 5,
        calls: AtomicUsize::new(0),
    }));
    let chat_body = |request_id: &str, user_content: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-request-id", request_id)
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a coding agent."},
                        {"role": "user", "content": user_content}
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup project context.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }],
                    "max_tokens": 8
                })
                .to_string(),
            ))
            .expect("request builds")
    };

    let first = app
        .clone()
        .oneshot(chat_body("agent-turn-1", "first turn"))
        .await
        .expect("first chat response");
    assert_eq!(first.status(), StatusCode::OK);
    let second = app
        .clone()
        .oneshot(chat_body("agent-turn-2", "second turn"))
        .await
        .expect("second chat response");
    assert_eq!(second.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 2);
    let cold = &recent[0];
    let warm = &recent[1];

    assert_eq!(cold["request_id"], "agent-turn-1");
    assert_eq!(cold["cache_status"], "miss");
    assert_eq!(cold["cached_tokens"], 0);
    assert_eq!(cold["uncached_tokens"], 100);
    assert_eq!(warm["request_id"], "agent-turn-2");
    assert_eq!(warm["cache_status"], "hit");
    assert_eq!(warm["cached_tokens"], 100);
    assert_eq!(warm["uncached_tokens"], 0);
    assert_eq!(cold["cache_template_id"], "chatml/qwen/v1");
    assert_eq!(cold["model_family"], "qwen");
    assert_eq!(cold["stable_prefix_key"], warm["stable_prefix_key"]);
    assert_ne!(cold["prompt_hash"], warm["prompt_hash"]);
    assert!(
        cold["cache_key"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["prompt_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["tool_schema_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["system_prompt_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
}

#[tokio::test]
async fn admin_metrics_request_cache_derives_repeat_prefix_status_without_cached_token_usage() {
    let app = build_router_with_unauthenticated_admin(Box::new(CachedUsageBackend {
        prompt_tokens: 100,
        prompt_cached_tokens: None,
        completion_tokens: 5,
    }));
    let chat_body = |request_id: &str, user_content: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-request-id", request_id)
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a coding agent."},
                        {"role": "user", "content": user_content}
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup project context.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }],
                    "max_tokens": 8
                })
                .to_string(),
            ))
            .expect("request builds")
    };

    let first = app
        .clone()
        .oneshot(chat_body("agent-fallback-1", "first turn"))
        .await
        .expect("first chat response");
    assert_eq!(first.status(), StatusCode::OK);
    let second = app
        .clone()
        .oneshot(chat_body("agent-fallback-2", "second turn"))
        .await
        .expect("second chat response");
    assert_eq!(second.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 2);
    let cold = &recent[0];
    let warm = &recent[1];

    assert_eq!(cold["request_id"], "agent-fallback-1");
    assert_eq!(cold["cache_status"], "unknown");
    assert!(cold["cached_tokens"].is_null());
    assert!(cold["uncached_tokens"].is_null());
    assert_eq!(warm["request_id"], "agent-fallback-2");
    assert_eq!(warm["cache_status"], "partial");
    assert!(warm["cached_tokens"].is_null());
    assert!(warm["uncached_tokens"].is_null());
    assert_eq!(cold["stable_prefix_key"], warm["stable_prefix_key"]);
    assert_ne!(cold["prompt_hash"], warm["prompt_hash"]);
}

#[tokio::test]
async fn admin_metrics_request_cache_records_streamed_observation() {
    let app = build_router_with_unauthenticated_admin(Box::new(CachedUsageBackend {
        prompt_tokens: 50,
        prompt_cached_tokens: Some(50),
        completion_tokens: 3,
    }));
    let request_id = "cache-streamed";
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-request-id", request_id)
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat stream response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("[DONE]"));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["streamed"], true);
    assert_eq!(observation["prompt_tokens"], 50);
    assert_eq!(observation["cached_tokens"], 50);
    assert_eq!(observation["uncached_tokens"], 0);
    assert_eq!(observation["cache_status"], "hit");
}
