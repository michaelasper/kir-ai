use super::*;

#[tokio::test]
async fn duplicate_active_request_id_fails_closed() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = build_router_with_backend_and_options(
        Box::new(BlockingBackend {
            entered: entered.clone(),
            release: release.clone(),
        }),
        EngineOptions {
            concurrency_limit: 2,
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let first = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("first", "same-id")),
    );
    entered.notified().await;

    let second = app
        .clone()
        .oneshot(chat_request_body_with_id("second", "same-id"))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::CONFLICT);
    let body = body_json(second.into_body()).await;
    assert_eq!(body["error"]["code"], "request_id_conflict");
    assert_eq!(body["error"]["phase"], "request_validation");

    release.notify_waiters();
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_returns_openai_shape() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
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
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], llm_engine::DEFAULT_MODEL_ID);
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_ascii_lowercase()
            .contains("rust")
    );
}

#[tokio::test]
async fn chat_completions_protocol_test_backend_is_not_fake_chat_inference() {
    let content = protocol_test_chat_content(json!([
        {
            "role": "user",
            "content": "Use codename Saffron-42 and build color teal. What codename and color should you remember?"
        }
    ]))
    .await;
    assert_eq!(content, "hello from rust native backend");
}

#[tokio::test]
async fn chat_completions_adapts_protocol_response_across_turns() {
    let poem = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short, vivid poem about dogs."}
    ]))
    .await;
    let critique = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short, vivid poem about dogs."},
        {"role": "assistant", "content": poem},
        {"role": "user", "content": "Critique the poem with concrete feedback."}
    ]))
    .await;
    let rewrite = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short, vivid poem about dogs."},
        {"role": "assistant", "content": poem},
        {"role": "user", "content": "Critique the poem with concrete feedback."},
        {"role": "assistant", "content": critique},
        {"role": "user", "content": "Rewrite the poem applying that feedback."}
    ]))
    .await;

    assert_ne!(poem, critique);
    assert_ne!(critique, rewrite);
    assert!(poem.to_ascii_lowercase().contains("dog"));
    assert!(critique.to_ascii_lowercase().contains("feedback"));
    assert!(rewrite.to_ascii_lowercase().contains("revised"));
}

#[tokio::test]
async fn chat_completions_revises_poem_from_feedback_without_repeating_original() {
    let original = "Dogs flash through rain-wet grass, brave hearts chasing the sun.";
    let revised = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short poem about dogs."},
        {"role": "assistant", "content": original},
        {
            "role": "user",
            "content": "Feedback: The image is lively, but it is only one sentence and feels generic. Please revise it into four short lines with a clearer rhythm, more concrete dog details like paws or tails, and a warmer emotional turn. Avoid vague phrases like brave hearts."
        }
    ]))
    .await;

    assert_ne!(revised, original);
    assert!(!revised.to_ascii_lowercase().contains("brave hearts"));
    assert!(
        revised
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count()
            >= 4,
        "revised poem should use multiple short lines: {revised}"
    );
    let lower = revised.to_ascii_lowercase();
    assert!(lower.contains("paws") || lower.contains("tails"));
}

#[tokio::test]
async fn chat_completions_handles_open_ended_multi_turn_story_fixture() {
    let story = protocol_chat_content(json!([
        {"role": "user", "content": "Write a tiny children's story about a shy dog named Miso."}
    ]))
    .await;
    assert!(story.contains("Miso"));
    assert!(!story.contains("hello from rust native backend"));

    let specific = protocol_chat_content(json!([
        {"role": "user", "content": "Write a tiny children's story about a shy dog named Miso."},
        {"role": "assistant", "content": story},
        {"role": "user", "content": "Make it more specific with one toy and one place."}
    ]))
    .await;
    assert!(specific.contains("Miso"));
    assert!(specific.contains("sock") || specific.contains("porch"));

    let bullets = protocol_chat_content(json!([
        {"role": "user", "content": "Write a tiny children's story about a shy dog named Miso."},
        {"role": "assistant", "content": story},
        {"role": "user", "content": "Make it more specific with one toy and one place."},
        {"role": "assistant", "content": specific},
        {"role": "user", "content": "Explain in two bullets what changed."}
    ]))
    .await;
    assert!(
        bullets
            .lines()
            .filter(|line| line.starts_with("- "))
            .count()
            >= 2
    );

    let bedtime = protocol_chat_content(json!([
        {"role": "user", "content": "Write a tiny children's story about a shy dog named Miso."},
        {"role": "assistant", "content": story},
        {"role": "user", "content": "Make it more specific with one toy and one place."},
        {"role": "assistant", "content": specific},
        {"role": "user", "content": "Explain in two bullets what changed."},
        {"role": "assistant", "content": bullets},
        {"role": "user", "content": "Revise it as a quiet bedtime version."}
    ]))
    .await;
    assert!(bedtime.contains("Miso"));
    assert!(bedtime.to_ascii_lowercase().contains("bedtime"));

    let memory = protocol_chat_content(json!([
        {"role": "user", "content": "Write a tiny children's story about a shy dog named Miso."},
        {"role": "assistant", "content": story},
        {"role": "user", "content": "Make it more specific with one toy and one place."},
        {"role": "assistant", "content": specific},
        {"role": "user", "content": "Revise it as a quiet bedtime version."},
        {"role": "assistant", "content": bedtime},
        {"role": "user", "content": "Memory check: what is the dog's name?"}
    ]))
    .await;
    assert!(memory.contains("Miso"));
}

#[tokio::test]
async fn chat_completions_handles_dog_poem_follow_up_turns() {
    let original = "Dogs flash through rain-wet grass, brave hearts chasing the sun.";
    let revised = "Revised poem:\nPaws tap softly by the door,\nTails sweep circles on the floor,\nWarm noses nudge the evening in,\nHome begins where dogs have been.";

    let explanation = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short poem about dogs."},
        {"role": "assistant", "content": original},
        {"role": "user", "content": "Feedback: The image is lively, but it is only one sentence and feels generic. Please revise it into four short lines with a clearer rhythm, more concrete dog details like paws or tails, and a warmer emotional turn. Avoid vague phrases like brave hearts."},
        {"role": "assistant", "content": revised},
        {"role": "user", "content": "Explain what changed in the revision."}
    ]))
    .await;
    assert_ne!(explanation, revised);
    assert!(explanation.to_ascii_lowercase().contains("changed"));

    let bedtime = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short poem about dogs."},
        {"role": "assistant", "content": original},
        {"role": "user", "content": "Feedback: The image is lively, but it is only one sentence and feels generic. Please revise it into four short lines with a clearer rhythm, more concrete dog details like paws or tails, and a warmer emotional turn. Avoid vague phrases like brave hearts."},
        {"role": "assistant", "content": revised},
        {"role": "user", "content": "Revise again into a quieter bedtime version."}
    ]))
    .await;
    assert_ne!(bedtime, revised);
    assert!(bedtime.to_ascii_lowercase().contains("bedtime"));

    let memory = protocol_chat_content(json!([
        {"role": "user", "content": "Write a short poem about dogs."},
        {"role": "assistant", "content": original},
        {"role": "user", "content": "Feedback: The image is lively, but it is only one sentence and feels generic. Please revise it into four short lines with a clearer rhythm, more concrete dog details like paws or tails, and a warmer emotional turn. Avoid vague phrases like brave hearts."},
        {"role": "assistant", "content": revised},
        {"role": "user", "content": "Memory check: what phrase did we avoid?"}
    ]))
    .await;
    assert!(memory.to_ascii_lowercase().contains("brave hearts"));
}

#[tokio::test]
async fn chat_completions_accepts_text_content_parts() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "text", "text": " world"}
                            ]
                        }]
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
}
