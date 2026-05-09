use super::*;

#[test]
fn classifies_high_output_empty_completion_as_no_progress() {
    let class = llm_runtime::classify_no_progress("", 4096, false);
    assert_eq!(class, Some(NoProgressClass::EmptyHighOutputCompletion));
}

#[test]
fn content_delta_is_progress_even_with_many_tokens() {
    let class = llm_runtime::classify_no_progress("patched Cargo.toml", 4096, false);
    assert_eq!(class, None);
}

#[tokio::test]
async fn no_progress_transcript_replay_fixtures_return_stable_codes() {
    for fixture_json in [
        include_str!("../fixtures/no_progress/hidden_only_reasoning.json"),
        include_str!("../fixtures/no_progress/repeated_invalid_tool_call.json"),
        include_str!("../fixtures/no_progress/repeated_assistant_content.json"),
        include_str!("../fixtures/no_progress/stalled_assistant_turn.json"),
    ] {
        let fixture: Value = serde_json::from_str(fixture_json).expect("fixture json parses");
        let request = serde_json::from_value::<ChatCompletionRequest>(
            fixture.get("request").expect("fixture has request").clone(),
        )
        .expect("fixture request parses");
        let runtime = Runtime::new(ReplayBackend {
            output: fixture_backend_output(
                fixture
                    .get("backend_output")
                    .expect("fixture has backend output"),
            ),
        });

        let err = runtime
            .chat(request)
            .await
            .expect_err("fixture must replay no-progress failure");
        let RuntimeError::NoProgress(class) = err else {
            panic!("expected no-progress error for fixture {fixture:?}");
        };
        assert_eq!(
            class.code(),
            fixture["expected_code"]
                .as_str()
                .expect("fixture has expected code")
        );
    }
}
