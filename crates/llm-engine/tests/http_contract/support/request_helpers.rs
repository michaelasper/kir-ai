async fn protocol_chat_content(messages: Value) -> String {
    let response = build_router_with_backend(Box::new(ScriptedChatBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": messages
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    body["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content")
        .to_owned()
}

async fn protocol_test_chat_content(messages: Value) -> String {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": messages
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    body["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content")
        .to_owned()
}

fn scripted_chat_response(prompt: &str) -> String {
    let current = last_user_message(prompt).to_ascii_lowercase();
    let prompt = prompt.to_ascii_lowercase();
    if prompt.contains("miso") {
        if current.contains("memory check") || current.contains("dog's name") {
            return "The dog's name is Miso.".to_owned();
        }
        if current.contains("bedtime") || current.contains("quiet") {
            return "Bedtime version: Miso curled beside the blue sock on the porch, listened to the moon, and fell asleep knowing the house was kind.".to_owned();
        }
        if current.contains("bullet") || current.contains("explain") {
            return "- I kept Miso as the shy dog so the thread has continuity.\n- I added the blue sock and porch so the story has concrete details.".to_owned();
        }
        if current.contains("specific") || current.contains("toy") || current.contains("place") {
            return "Miso carried a blue sock to the porch, peeked at the rain, and wagged when a child sat beside him.".to_owned();
        }
        if current.contains("story") {
            return "Miso was a shy little dog who hid behind a chair until a kind child offered a quiet hello.".to_owned();
        }
    }
    if current.contains("memory check") && prompt.contains("brave hearts") {
        "The avoided phrase was \"brave hearts.\"".to_owned()
    } else if current.contains("explain") && prompt.contains("brave hearts") {
        "The revision changed the one-sentence image into short lines, replaced brave hearts with paws and tails, and made the ending warmer.".to_owned()
    } else if current.contains("bedtime") && prompt.contains("dog") {
        "Bedtime version:\nSoft paws settle by the bed,\nSleepy tails make one last sweep,\nWarm noses rest near open hands,\nDogs turn the quiet house to sleep.".to_owned()
    } else if (current.contains("rewrite")
        || current.contains("revise")
        || current.contains("revised"))
        && prompt.contains("feedback")
    {
        "Revised poem:\nPaws tap softly by the door,\nTails sweep circles on the floor,\nWarm noses nudge the evening in,\nHome begins where dogs have been.".to_owned()
    } else if current.contains("critique") && current.contains("feedback") {
        "Feedback: The dog poem has clear motion; add sharper images and a stronger final line."
            .to_owned()
    } else if current.contains("poem") && current.contains("dog") {
        "Dogs flash through rain-wet grass, brave hearts chasing the sun.".to_owned()
    } else {
        "Unsupported scripted chat test prompt.".to_owned()
    }
}

fn last_user_message(prompt: &str) -> String {
    const USER_START: &str = "<|im_start|>user\n";
    let Some(start) = prompt.rfind(USER_START) else {
        return prompt.to_owned();
    };
    let body_start = start + USER_START.len();
    let body = &prompt[body_start..];
    let end = body.find("<|im_end|>").unwrap_or(body.len());
    body[..end].to_owned()
}

fn test_token_count(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

async fn body_json(body: Body) -> Value {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}

async fn body_text(body: Body) -> String {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    String::from_utf8(bytes.to_vec()).expect("utf8 body")
}

fn chat_request_body(content: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "messages": [{"role": "user", "content": content}]
            })
            .to_string(),
        ))
        .expect("request builds")
}

fn chat_request_body_with_id(content: &str, request_id: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "messages": [{"role": "user", "content": content}]
            })
            .to_string(),
        ))
        .expect("request builds")
}

fn completion_request_body_with_id(prompt: &str, request_id: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "prompt": prompt
            })
            .to_string(),
        ))
        .expect("request builds")
}
