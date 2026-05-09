use llm_backend::ProtocolTestBackend;

pub(super) fn protocol_test_backend() -> ProtocolTestBackend {
    ProtocolTestBackend::new("local-qwen36", "hello from rust native backend")
        .with_required_tool_protocol()
        .with_json_object_protocol()
}
