use super::*;

#[test]
fn engine_options_reject_remote_http_hub_endpoint_with_token() {
    let result = build_router_with_backend_and_options(
        Box::new(llm_backend::ProtocolTestBackend::new("local-qwen36", "ok")),
        EngineOptions {
            hub_endpoint: Some("http://example.com".to_owned()),
            hf_token: Some("hf_secret".to_owned()),
            ..EngineOptions::default()
        },
    );

    let err = match result {
        Ok(_) => panic!("remote HTTP endpoint with HF_TOKEN should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("refusing to send HF_TOKEN to non-HTTPS hub endpoint"),
        "error: {err}"
    );
}

#[test]
fn engine_options_allow_loopback_http_hub_endpoint_with_token() {
    let result = build_router_with_backend_and_options(
        Box::new(llm_backend::ProtocolTestBackend::new("local-qwen36", "ok")),
        EngineOptions {
            hub_endpoint: Some("http://127.0.0.1:8080".to_owned()),
            hf_token: Some("hf_secret".to_owned()),
            ..EngineOptions::default()
        },
    );

    assert!(result.is_ok());
}
