use llm_tokenizer::HuggingFaceTokenizer;

const TOKENIZER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/qwen36/tokenizer.json"
);

#[test]
fn official_qwen36_tokenizer_round_trips_text() {
    let tokenizer =
        HuggingFaceTokenizer::from_file(TOKENIZER_PATH).expect("official tokenizer loads");

    let ids = tokenizer
        .encode("hello rust tokenizer", false)
        .expect("text encodes");
    assert!(!ids.is_empty());

    let decoded = tokenizer.decode(&ids, false).expect("text decodes");
    assert_eq!(decoded, "hello rust tokenizer");
}

#[test]
fn official_qwen36_tokenizer_preserves_chatml_special_tokens() {
    let tokenizer =
        HuggingFaceTokenizer::from_file(TOKENIZER_PATH).expect("official tokenizer loads");

    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(248_045));
    assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(248_046));

    let ids = tokenizer
        .encode("<|im_start|>user\nhi<|im_end|>\n", false)
        .expect("chatml encodes");
    assert!(ids.contains(&248_045));
    assert!(ids.contains(&248_046));
}
