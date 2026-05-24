use super::*;

#[test]
fn stop_tokens_match_literal_ids_and_tokenizer_tokens() {
    let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/qwen36/tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads");
    let im_end = tokenizer
        .token_to_id("<|im_end|>")
        .expect("qwen tokenizer has im_end token") as usize;
    let stop_tokens = NativeTextStopTokens {
        token_ids: &[1],
        token_strings: &["<|im_end|>"],
        encoded_token_strings: &[],
    };
    let non_stop = (0..16)
        .find(|token_id| *token_id != 1 && *token_id != im_end)
        .expect("small non-stop token id exists");

    let resolved = stop_tokens.resolve(&tokenizer);

    assert_eq!(resolved.token_ids(), vec![1, im_end]);
    assert!(resolved.contains(1));
    assert!(resolved.contains(im_end));
    assert!(!resolved.contains(non_stop));
    assert!(
        !resolved.contains(im_end + (u32::MAX as usize) + 1),
        "candidate ids above u32::MAX must not wrap into a tokenizer stop token"
    );

    const PHRASE_STOP_STRINGS: &[&str] = &["hello rust tokenizer"];
    let missing_literal = PHRASE_STOP_STRINGS[0];
    assert!(
        tokenizer.token_to_id(missing_literal).is_none(),
        "fixture phrase should not be treated as a single literal stop token"
    );
    let literal_only_stop_tokens = NativeTextStopTokens {
        token_ids: &[],
        token_strings: PHRASE_STOP_STRINGS,
        encoded_token_strings: &[],
    }
    .resolve(&tokenizer);
    assert_eq!(literal_only_stop_tokens.token_ids(), Vec::<usize>::new());

    let phrase = PHRASE_STOP_STRINGS[0];
    assert!(
        tokenizer.token_to_id(phrase).is_none(),
        "fixture phrase should exercise encode fallback for non-vocabulary stop strings"
    );
    let phrase_ids = tokenizer
        .encode(phrase, false)
        .expect("fixture phrase encodes");
    let phrase_stop_tokens = NativeTextStopTokens {
        token_ids: &[],
        token_strings: &[],
        encoded_token_strings: PHRASE_STOP_STRINGS,
    }
    .resolve(&tokenizer);
    for token_id in phrase_ids {
        assert!(phrase_stop_tokens.contains(token_id as usize));
    }
}

#[test]
fn driver_stop_token_candidate_is_not_emitted_for_blocking_generation() {
    let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
        NativeTextStopTokens {
            token_ids: &[1],
            token_strings: &[],
            encoded_token_strings: &[],
        },
    ));

    let output = driver
        .generate_blocking(driver_test_request(4), CancellationToken::new())
        .expect("generation stops cleanly");

    assert_eq!(output.text, "");
    assert_eq!(output.completion_tokens, 0);
    assert_eq!(output.finish_reason, BackendFinishReason::Stop);
}

#[test]
fn driver_stop_token_candidate_is_not_emitted_for_streaming_generation() {
    let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
        NativeTextStopTokens {
            token_ids: &[1],
            token_strings: &[],
            encoded_token_strings: &[],
        },
    ));
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);

    driver
        .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
        .expect("streaming generation stops cleanly");
    let final_chunk = loop {
        let chunk = rx
            .blocking_recv()
            .expect("final chunk is sent")
            .expect("final chunk is ok");
        if chunk.finish_reason.is_some() {
            break chunk;
        }
        assert_eq!(chunk.text, "");
        assert_eq!(chunk.completion_tokens, 0);
    };
    assert_eq!(final_chunk.text, "");
    assert_eq!(final_chunk.completion_tokens, 0);
    assert_eq!(final_chunk.finish_reason, Some(BackendFinishReason::Stop));
    assert!(rx.blocking_recv().is_none());
}

#[test]
fn streaming_generation_decodes_each_output_token_once() {
    let adapter = TestAdapter::new([1_usize, 2, 3, 4]);
    let full_decode_token_total = adapter.decoded_token_total();
    let stream_decoded_token_total = adapter.stream_decoded_token_total();
    let driver = driver_for_test(adapter);
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);

    driver
        .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
        .expect("streaming generation succeeds");

    let mut text = String::new();
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk is ok");
        text.push_str(&chunk.text);
    }

    assert_eq!(text, "<1><2><3><4>");
    assert_eq!(stream_decoded_token_total.load(Ordering::SeqCst), 4);
    assert_eq!(full_decode_token_total.load(Ordering::SeqCst), 0);
}

#[test]
fn streaming_generation_preserves_unicode_token_boundaries() {
    let driver = driver_for_test(TestAdapter::new([1_usize, 2]).with_unicode_boundary_decode());
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);

    driver
        .generate_blocking_stream(driver_test_request(2), tx, CancellationToken::new())
        .expect("streaming generation succeeds");

    let mut text = String::new();
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk is ok");
        text.push_str(&chunk.text);
    }

    assert_eq!(text, "é");
}

#[test]
fn streaming_generation_decode_work_scales_with_output_tokens() {
    let script = (1_usize..=64).collect::<Vec<_>>();
    let adapter = TestAdapter::new(std::sync::Arc::<[usize]>::from(script.clone()))
        .with_max_position_embeddings(128);
    let stream_decoded_token_total = adapter.stream_decoded_token_total();
    let full_decode_token_total = adapter.decoded_token_total();
    let driver = driver_for_test(adapter).with_max_new_tokens(64);
    let (tx, mut rx) = tokio::sync::mpsc::channel(128);

    driver
        .generate_blocking_stream(driver_test_request(64), tx, CancellationToken::new())
        .expect("streaming generation succeeds");

    let mut text = String::new();
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk is ok");
        text.push_str(&chunk.text);
    }

    let expected = script
        .iter()
        .map(|token_id| format!("<{token_id}>"))
        .collect::<String>();
    assert_eq!(text, expected);
    assert_eq!(stream_decoded_token_total.load(Ordering::SeqCst), 64);
    assert_eq!(full_decode_token_total.load(Ordering::SeqCst), 0);
}

#[test]
fn driver_supplies_rng_draws_only_for_non_greedy_sampling() {
    let greedy_adapter = TestAdapter::new([1_usize, 2]);
    let greedy_draws = greedy_adapter.sampling_draws();
    let greedy_driver = driver_for_test(greedy_adapter);

    greedy_driver
        .generate_blocking(driver_test_request(2), CancellationToken::new())
        .expect("greedy generation succeeds");

    assert_eq!(
        *greedy_draws
            .lock()
            .expect("greedy sampling draws lock is not poisoned"),
        vec![None, None]
    );

    let top_p_adapter = TestAdapter::new([1_usize, 2]);
    let top_p_draws = top_p_adapter.sampling_draws();
    let top_p_driver = driver_for_test(top_p_adapter);
    let mut request = driver_test_request(2);
    request.sampling = SamplingConfig::TopP {
        temperature: 1.0,
        top_p: 0.9,
    };

    top_p_driver
        .generate_blocking(request, CancellationToken::new())
        .expect("top-p generation succeeds");

    let top_p_draws = top_p_draws
        .lock()
        .expect("top-p sampling draws lock is not poisoned")
        .clone();
    assert_eq!(top_p_draws.len(), 2);
    assert!(
        top_p_draws
            .into_iter()
            .all(|draw| { matches!(draw, Some(value) if (0.0..1.0).contains(&value)) })
    );
}
