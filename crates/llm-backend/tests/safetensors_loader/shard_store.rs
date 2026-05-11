use super::*;

#[test]
fn shard_store_reads_bf16_row_by_tensor_name() {
    let root = temp_snapshot_dir("indexed-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_shard_count(), 0);
    assert_eq!(
        store.tensor_names().collect::<Vec<_>>(),
        vec!["embed.weight"]
    );

    assert_eq!(
        store.bf16_row_f32("embed.weight", 1).expect("row"),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        store
            .bf16_tensor_bits_range("embed.weight", 3, 2)
            .expect("raw bf16 range"),
        vec![bf16_bits(4.0), bf16_bits(5.0)]
    );
    assert_eq!(store.cached_shard_count(), 1);
    assert_eq!(
        store
            .bf16_matvec_row_major_f32("embed.weight", &[1.0, 2.0, 3.0])
            .expect("matvec"),
        vec![14.0, 32.0]
    );
    assert_eq!(
        store
            .bf16_matvecs_row_major_f32("embed.weight", &[vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]])
            .expect("batched matvec"),
        vec![vec![14.0, 32.0], vec![10.0, 28.0]]
    );
    let top = store
        .bf16_matvec_top_k_rows_f32("embed.weight", &[1.0, 2.0, 3.0], 1, 1)
        .expect("top logits");
    assert_eq!(top[0].index, 1);
    assert_eq!(top[0].logit, 32.0);
    assert_eq!(
        store
            .tensor_shard_path("embed.weight")
            .expect("shard path")
            .file_name()
            .and_then(|name| name.to_str()),
        Some("model-00001-of-00001.safetensors")
    );
    assert_eq!(store.cached_shard_count(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_materializes_shard_once_and_reuses_it_for_reads() {
    let root = temp_snapshot_dir("materialized-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    let shard_bytes =
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let shard_len = shard_bytes.len();
    std::fs::write(root.join("model-00001-of-00001.safetensors"), shard_bytes).expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.materialized_shard_count(), 0);

    assert_eq!(
        store
            .materialize_shard_for_tensor("embed.weight")
            .expect("materialized shard"),
        shard_len
    );
    assert_eq!(store.materialized_shard_count(), 1);
    assert_eq!(
        store
            .materialize_shard_for_tensor("embed.weight")
            .expect("reused materialized shard"),
        shard_len
    );
    assert_eq!(store.materialized_shard_count(), 1);
    assert_eq!(
        store
            .bf16_tensor_f32_range("embed.weight", 1, 4)
            .expect("range reads from materialized shard"),
        vec![2.0, 3.0, 4.0, 5.0]
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_materializes_all_indexed_shards_once() {
    let root = temp_snapshot_dir("materialized-all-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": {
                "embed.weight": "embed.safetensors",
                "norm.weight": "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    let embed = tiny_safetensors_bf16("embed.weight", &[2], &[1.0, 2.0]);
    let norm = tiny_safetensors_bf16("norm.weight", &[2], &[3.0, 4.0]);
    let expected_bytes = embed.len() + norm.len();
    std::fs::write(root.join("embed.safetensors"), embed).expect("embed shard");
    std::fs::write(root.join("norm.safetensors"), norm).expect("norm shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_shard_count(), 0);
    assert_eq!(store.materialized_shard_count(), 0);

    assert_eq!(
        store
            .materialize_all_shards()
            .expect("all shards materialize"),
        expected_bytes
    );
    assert_eq!(store.cached_shard_count(), 2);
    assert_eq!(store.materialized_shard_count(), 2);
    assert_eq!(
        store
            .materialize_all_shards()
            .expect("materialized shards are reused"),
        expected_bytes
    );
    assert_eq!(store.materialized_shard_count(), 2);
    assert_eq!(
        store.bf16_tensor_f32("norm.weight").expect("read norm"),
        vec![3.0, 4.0]
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_rejects_unsafe_index_shard_paths_on_open() {
    for shard_path in [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "nested\\outside.safetensors",
    ] {
        let root = temp_snapshot_dir(&format!("unsafe-index-{}", shard_path.len()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).expect("snapshot dir");
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 2 },
                "weight_map": { "embed.weight": shard_path }
            })
            .to_string(),
        )
        .expect("index");

        let err = SafeTensorShardStore::open(&root).expect_err("unsafe index fails closed");

        assert_eq!(err.code(), "model_integrity_failed");
        std::fs::remove_dir_all(root).ok();
    }
}

#[cfg(unix)]
#[test]
fn shard_store_rejects_symlink_that_escapes_snapshot_root() {
    let root = temp_snapshot_dir("symlink-escape");
    let outside = temp_safetensors_path("symlink-outside");
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_file(&outside).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 2 },
            "weight_map": { "embed.weight": "linked.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        &outside,
        tiny_safetensors_bf16("embed.weight", &[1], &[1.0]),
    )
    .expect("outside shard");
    std::os::unix::fs::symlink(&outside, root.join("linked.safetensors")).expect("escape symlink");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let err = store
        .bf16_tensor_f32("embed.weight")
        .expect_err("escaped symlink fails closed");

    assert_eq!(err.code(), "model_integrity_failed");
    std::fs::remove_dir_all(root).ok();
    std::fs::remove_file(outside).ok();
}

#[test]
fn f32_range_cached_returns_same_values_as_uncached() {
    let root = temp_snapshot_dir("f32-cache-values");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_f32_count(), 0);

    let uncached = store
        .bf16_tensor_f32_range("embed.weight", 0, 3)
        .expect("uncached read");
    let cached = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 3)
        .expect("cached read");
    assert_eq!(uncached, cached);
    assert_eq!(store.cached_f32_count(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn f32_range_cached_populates_on_first_access_and_hits_on_second() {
    let root = temp_snapshot_dir("f32-cache-hits");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_f32_count(), 0);

    let first = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("first cached read");
    assert_eq!(store.cached_f32_count(), 1);

    let second = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("second cached read");
    assert_eq!(store.cached_f32_count(), 1);
    assert_eq!(first, second);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn f32_cached_reads_full_tensor_and_caches_it() {
    let root = temp_snapshot_dir("f32-cache-full");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": { "norm.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("norm.weight", &[2], &[3.0, 4.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_f32_count(), 0);

    let first = store
        .bf16_tensor_f32_cached("norm.weight")
        .expect("first full cached");
    assert_eq!(store.cached_f32_count(), 1);

    let second = store
        .bf16_tensor_f32_cached("norm.weight")
        .expect("second full cached");
    assert_eq!(store.cached_f32_count(), 1);
    assert_eq!(first, second);
    assert_eq!(first, vec![3.0, 4.0]);
    std::fs::remove_dir_all(root).ok();
}
