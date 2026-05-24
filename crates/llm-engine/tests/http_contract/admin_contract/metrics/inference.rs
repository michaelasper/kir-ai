use super::*;

#[tokio::test]
async fn admin_metrics_report_inference_counts_and_tokens() {
    let app = build_router_with_protocol_test_backend();
    let response = app
        .clone()
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
    assert_eq!(body["requests_total"], 1);
    assert_eq!(body["successful_requests"], 1);
    assert_eq!(body["failed_requests"], 0);
    assert_eq!(body["streamed_requests"], 0);
    assert_eq!(body["stream_client_disconnected_requests"], 0);
    assert_eq!(body["stream_stalled_requests"], 0);
    assert_eq!(body["tokens"]["prompt_tokens"], 1);
    assert!(
        body["tokens"].get("prompt_tokens_details").is_none(),
        "cached prompt token details should be absent when no successful request reported them"
    );
    let completion_tokens = body["tokens"]["completion_tokens"]
        .as_u64()
        .expect("completion tokens are numeric");
    assert!(completion_tokens > 0);
    assert_eq!(body["tokens"]["total_tokens"], completion_tokens + 1);
    assert_eq!(body["request_latency_ms"]["count"], 1);
    assert_eq!(body["non_streamed_request_latency_ms"]["count"], 1);
    assert_eq!(body["streamed_request_latency_ms"]["count"], 0);
    assert!(
        body["request_latency_ms"]["max"]
            .as_f64()
            .expect("latency max is numeric")
            >= body["request_latency_ms"]["min"]
                .as_f64()
                .expect("latency min is numeric")
    );
    assert!(
        body["tokens_per_second"]
            .as_f64()
            .expect("tokens per second is numeric")
            > 0.0
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kernels"].is_object(),
        "native text Metal metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kv_cache"]["f16_resident_bytes"].is_number(),
        "native text Metal f16 KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kv_cache"]["int8_resident_bytes"].is_number(),
        "native text Metal int8 KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kv_cache"]["stage_resident_bytes"]
            .is_number(),
        "native text Metal staged KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kv_cache"]["stage_bytes_copied"].is_number(),
        "native text Metal staged KV cache copy metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hits"].is_number(),
        "native text Qwen prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["entries_scanned"].is_number(),
        "native text Qwen prefix cache lookup scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["namespace_entries_scanned"]
            .is_number(),
        "native text Qwen prefix cache namespace scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hit_clone_bytes"].is_number(),
        "native text Qwen prefix cache hit clone bytes are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["prefill_chunks"].is_number(),
        "native text Qwen prefix cache prefill chunk metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["prefill_tokens"].is_number(),
        "native text Qwen prefix cache prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["checkpoint_stores"]
            .is_number(),
        "native text Qwen prefix cache checkpoint store metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["checkpoint_reuse_hits"]
            .is_number(),
        "native text Qwen prefix cache checkpoint reuse metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["shared_prefix_hits"]
            .is_number(),
        "native text Qwen shared-prefix hit metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["shared_prefix_reused_tokens"]
            .is_number(),
        "native text Qwen shared-prefix token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hit_tokens"].is_number(),
        "native text Qwen prefix cache hit token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["miss_tokens"].is_number(),
        "native text Qwen prefix cache miss token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["avoided_prefill_tokens"]
            .is_number(),
        "native text Qwen prefix cache avoided prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hits"].is_number(),
        "native text Gemma prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["entries_scanned"].is_number(),
        "native text Gemma prefix cache lookup scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["namespace_entries_scanned"]
            .is_number(),
        "native text Gemma prefix cache namespace scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hit_clone_bytes"].is_number(),
        "native text Gemma prefix cache hit clone bytes are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["prefill_chunks"].is_number(),
        "native text Gemma prefix cache prefill chunk metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["prefill_tokens"].is_number(),
        "native text Gemma prefix cache prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["checkpoint_stores"]
            .is_number(),
        "native text Gemma prefix cache checkpoint store metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["checkpoint_reuse_hits"]
            .is_number(),
        "native text Gemma prefix cache checkpoint reuse metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["shared_prefix_hits"]
            .is_number(),
        "native text Gemma shared-prefix hit metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["shared_prefix_reused_tokens"]
            .is_number(),
        "native text Gemma shared-prefix token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hit_tokens"].is_number(),
        "native text Gemma prefix cache hit token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["miss_tokens"].is_number(),
        "native text Gemma prefix cache miss token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["avoided_prefill_tokens"]
            .is_number(),
        "native text Gemma prefix cache avoided prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["requests_total"].is_number(),
        "MLX sidecar request metrics are exposed"
    );
    assert!(body["backend_metrics"]["mlx"]["successful_requests"].is_number());
    assert!(body["backend_metrics"]["mlx"]["failed_requests"].is_number());
    assert!(body["backend_metrics"]["mlx"]["stream_chunks"].is_number());
    assert!(
        body["backend_metrics"]["mlx"]["request_latency_ms"]["count"].is_number(),
        "MLX sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["upstream_request_latency_ms"]["count"].is_number(),
        "MLX upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["blocking_upstream_request_latency_ms"]["count"].is_number(),
        "MLX blocking upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["streaming_upstream_request_latency_ms"]["count"]
            .is_number(),
        "MLX streaming upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kernels"].is_object(),
        "native Qwen Metal compatibility metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["bf16_matrix_cache"].is_object(),
        "native Qwen Metal BF16 matrix cache metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["bf16_matrix_cache"]["resident_bytes"]
            .is_number(),
        "native Qwen Metal BF16 matrix cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["resident_bytes"].is_number(),
        "native Qwen Metal KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["f16_resident_bytes"].is_number(),
        "native Qwen Metal f16 KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["int8_resident_bytes"].is_number(),
        "native Qwen Metal int8 KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["int8_bytes_uploaded"].is_number(),
        "native Qwen Metal int8 KV cache upload bytes are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["stage_resident_bytes"]
            .is_number(),
        "native Qwen Metal staged KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["stage_bytes_copied"].is_number(),
        "native Qwen Metal staged KV cache copy metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["linear_attention_cache"]["resident_bytes"]
            .is_number(),
        "native Qwen Metal linear cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"].is_object(),
        "native Qwen shared prefix cache metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hits"].is_number(),
        "native Qwen prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["misses"].is_number(),
        "native Qwen prefix cache misses are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["evictions"].is_number(),
        "native Qwen prefix cache evictions are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["resident_bytes"].is_number(),
        "native Qwen prefix cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["entries_scanned"].is_number(),
        "native Qwen prefix cache lookup scan metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["namespace_entries_scanned"]
            .is_number(),
        "native Qwen prefix cache namespace scan metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hit_clone_bytes"].is_number(),
        "native Qwen prefix cache hit clone bytes are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["prefill_chunks"].is_number(),
        "native Qwen prefix cache prefill chunk metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["prefill_tokens"].is_number(),
        "native Qwen prefix cache prefill token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["checkpoint_stores"].is_number(),
        "native Qwen prefix cache checkpoint store metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["checkpoint_reuse_hits"].is_number(),
        "native Qwen prefix cache checkpoint reuse metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["shared_prefix_hits"].is_number(),
        "native Qwen shared-prefix hit metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["shared_prefix_reused_tokens"]
            .is_number(),
        "native Qwen shared-prefix token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hit_tokens"].is_number(),
        "native Qwen prefix cache hit token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["miss_tokens"].is_number(),
        "native Qwen prefix cache miss token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["avoided_prefill_tokens"].is_number(),
        "native Qwen prefix cache avoided prefill token metrics are exposed through the legacy object"
    );
}
