use super::*;

#[tokio::test]
async fn qwen_moe_public_wrappers_route_and_run_selected_experts() {
    let root = temp_snapshot_dir("qwen-moe-public-wrappers");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_moe_forward_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();

    let router = qwen_layer0_moe_router(&store, &[2.0, 3.0], 2, &matvec)
        .await
        .expect("router");

    assert_eq!(router.selected[0].index, 1);
    assert_eq!(router.selected[1].index, 2);
    assert_close(
        &[router.selected[0].weight, router.selected[1].weight],
        &[0.7310586, 0.26894143],
        1e-6,
    );
    assert_eq!(matvec.softmax_top_k_calls(), 1);

    let dims = QwenMoeDims {
        hidden_size: 2,
        num_experts: 2,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
    };
    let selected = QwenMoeRouterProbe {
        logits: vec![0.0, 0.0],
        selected: vec![TopKWeight {
            index: 0,
            weight: 1.0,
        }],
    };

    let output = qwen_layer_moe_forward(&store, 0, &dims, &[1.0, 2.0], &selected, &matvec)
        .await
        .expect("moe forward");

    assert_close(&output, &[1.4621172, 2.9242344], 1e-6);
    assert_eq!(matvec.bf16_range_calls(), 2);
    std::fs::remove_dir_all(root).ok();
}
