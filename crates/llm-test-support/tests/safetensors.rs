use llm_test_support::safetensors::{
    TinySafetensorsSnapshot, tiny_safetensors_f32, tiny_safetensors_index_json,
};

#[test]
fn tiny_safetensors_f32_encodes_header_and_payload() {
    let bytes = tiny_safetensors_f32("linear.weight", &[1, 2], &[1.0, 2.0]);
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("header prefix")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header json");

    assert_eq!(header["linear.weight"]["dtype"], "F32");
    assert_eq!(header["linear.weight"]["shape"], serde_json::json!([1, 2]));
    assert_eq!(
        header["linear.weight"]["data_offsets"],
        serde_json::json!([0, 8])
    );
    assert_eq!(
        &bytes[8 + header_len..],
        [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat()
    );
}

#[test]
fn tiny_snapshot_builder_writes_index_and_shards() {
    let root =
        llm_test_support::safetensors::temp_snapshot_dir("llm-test-support", "snapshot-builder");
    std::fs::remove_dir_all(&root).ok();

    let written = TinySafetensorsSnapshot::new()
        .with_bf16_tensor("embed.safetensors", "embed.weight", [2], [1.0, 2.0])
        .with_bf16_tensor("norm.safetensors", "norm.weight", [1], [0.0])
        .write(&root)
        .expect("snapshot writes");

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("model.safetensors.index.json")).expect("index"),
    )
    .expect("index json");

    assert_eq!(written.tensor_count, 2);
    assert_eq!(written.shard_count, 2);
    assert_eq!(index["metadata"]["total_size"], written.total_size);
    assert_eq!(index["weight_map"]["embed.weight"], "embed.safetensors");
    assert_eq!(index["weight_map"]["norm.weight"], "norm.safetensors");
    assert!(root.join("embed.safetensors").is_file());
    assert!(root.join("norm.safetensors").is_file());
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn tiny_safetensors_index_json_maps_tensors_to_shards() {
    let index = tiny_safetensors_index_json(
        42,
        [
            ("embed.weight", "embed.safetensors"),
            ("norm.weight", "norm.safetensors"),
        ],
    );
    let index: serde_json::Value = serde_json::from_str(&index).expect("index json");

    assert_eq!(index["metadata"]["total_size"], 42);
    assert_eq!(index["weight_map"]["embed.weight"], "embed.safetensors");
    assert_eq!(index["weight_map"]["norm.weight"], "norm.safetensors");
}
