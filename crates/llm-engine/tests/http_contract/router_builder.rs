use super::*;

#[test]
fn public_router_builder_requires_explicit_backend() {
    let Err(err) = build_router() else {
        panic!("router builder without a backend must fail closed");
    };

    assert!(
        err.to_string().contains("explicit backend"),
        "error should explain how to provide a backend: {err}"
    );
}

#[test]
fn public_router_builders_with_backend_return_config_results() {
    let _: Result<Router, llm_engine::EngineConfigError> =
        llm_engine::router_builder(Box::new(FailingBackend)).build();
    let _: Result<Router, llm_engine::EngineConfigError> =
        llm_engine::router_builder(Box::new(FailingBackend))
            .with_concurrency(1)
            .build();
}
