use llm_backend::{
    GemmaLayerCache, NativeTextLayerCaches, SafeTensorShardStore, gemma_decode_token_with_cache,
    gemma_final_norm_for_spec, gemma_layer_caches_for_spec, gemma_lm_head_top_k_for_spec,
    gemma_prefill_sequence_with_cache,
    native_decode_token_with_cache as native_text_decode_token_with_cache,
    native_final_norm_for_spec as native_text_final_norm_for_spec,
    native_layer_caches_for_spec as native_text_layer_caches_for_spec,
    native_lm_head_top_k_for_spec as native_text_lm_head_top_k_for_spec,
    native_prefill_sequence_with_cache as native_text_prefill_sequence_with_cache,
};
use llm_models::{GemmaModelSpec, NativeTextModelSpec};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn gemma_layer_caches_cap_sliding_layers_to_sliding_window() {
    let spec = GemmaModelSpec::from_config_json(&tiny_gemma4_config(1, 8, &["sliding_attention"]))
        .expect("tiny Gemma config parses");

    let caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    assert_eq!(caches.len(), 1);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => {
            assert_eq!(cache.max_tokens(), 2);
            assert_eq!(cache.key_value_heads(), 1);
            assert_eq!(cache.head_dim(), 2);
        }
    }
}

#[test]
fn gemma_prefill_and_decode_produce_deterministic_tiny_outputs() {
    let root = temp_snapshot_dir("gemma-prefill-decode");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches).expect("prefill");
    assert_close(&prefill[0], &[2.0_f32.sqrt(), 0.0], 1e-5);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => assert_eq!(cache.token_count(), 2),
    }

    let decoded =
        gemma_decode_token_with_cache(&store, &spec, 2, &mut caches).expect("decode token");
    assert_close(&decoded, &[2.0 * 2.0_f32.sqrt(), 0.0], 1e-5);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => {
            assert_eq!(cache.token_count(), 2);
            assert_eq!(cache.next_position(), 3);
        }
    }

    std::fs::remove_dir_all(root).ok();
}

#[test]
fn gemma_prefill_supports_per_layer_inputs() {
    let root = temp_snapshot_dir("gemma-ple-prefill");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot_with_options(&root, true, 1, &["sliding_attention"], 0);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma PLE config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches).expect("prefill");

    assert!(spec.uses_per_layer_input());
    assert_close(&prefill[0], &[2.0 * 2.0_f32.sqrt(), 0.0], 1e-4);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn gemma_prefill_reuses_shared_kv_cache_layers() {
    let root = temp_snapshot_dir("gemma-shared-kv");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot_with_options(
        &root,
        false,
        2,
        &["sliding_attention", "sliding_attention"],
        1,
    );
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma shared KV config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches).expect("prefill");

    assert_eq!(caches.len(), 1);
    assert!(spec.is_kv_shared_layer(1));
    assert_close(&prefill[0], &[2.0_f32.sqrt(), 0.0], 1e-5);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn gemma_final_norm_and_tied_lm_head_select_top_token() {
    let root = temp_snapshot_dir("gemma-lm-head");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let final_norm =
        gemma_final_norm_for_spec(&store, &spec, &[2.0 * 2.0_f32.sqrt(), 0.0]).expect("norm");
    assert_close(&final_norm, &[2.0_f32.sqrt(), 0.0], 1e-5);
    let top = gemma_lm_head_top_k_for_spec(&store, &spec, &final_norm, 2, 64).expect("top logits");

    assert_eq!(top[0].index, 2);
    assert!((top[0].logit - 2.0 * 2.0_f32.sqrt()).abs() < 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn native_text_dispatch_matches_direct_gemma_prefill_decode_and_lm_head() {
    let root = temp_snapshot_dir("native-text-dispatch-gemma");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let native_spec = NativeTextModelSpec::Gemma(spec.clone());
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let mut direct_caches = gemma_layer_caches_for_spec(&spec, 8).expect("direct caches");
    let direct_prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut direct_caches)
            .expect("direct prefill");
    let direct_decode =
        gemma_decode_token_with_cache(&store, &spec, 2, &mut direct_caches).expect("direct decode");

    let mut native_caches =
        native_text_layer_caches_for_spec(&native_spec, 8).expect("native text caches");
    assert!(matches!(native_caches, NativeTextLayerCaches::Gemma(_)));
    let native_prefill =
        native_text_prefill_sequence_with_cache(&store, &native_spec, &[0, 1], &mut native_caches)
            .expect("native text prefill");
    let native_decode =
        native_text_decode_token_with_cache(&store, &native_spec, 2, &mut native_caches)
            .expect("native text decode");
    let native_norm =
        native_text_final_norm_for_spec(&store, &native_spec, &native_decode).expect("native norm");
    let native_top = native_text_lm_head_top_k_for_spec(&store, &native_spec, &native_norm, 2, 64)
        .expect("native top logits");

    assert_eq!(native_prefill.len(), direct_prefill.len());
    assert_close(&native_prefill[0], &direct_prefill[0], 1e-5);
    assert_close(&native_prefill[1], &direct_prefill[1], 1e-5);
    assert_close(&native_decode, &direct_decode, 1e-5);
    assert_eq!(native_top[0].index, 2);
    std::fs::remove_dir_all(root).ok();
}

fn write_tiny_gemma4_decoder_snapshot(root: &Path) {
    write_tiny_gemma4_decoder_snapshot_with_options(root, false, 1, &["sliding_attention"], 0);
}

fn write_tiny_gemma4_decoder_snapshot_with_options(
    root: &Path,
    per_layer_inputs: bool,
    num_hidden_layers: u32,
    layer_types: &[&str],
    num_kv_shared_layers: u32,
) {
    std::fs::write(
        root.join("config.json"),
        tiny_gemma4_config_with_options(
            num_hidden_layers,
            8,
            layer_types,
            per_layer_inputs,
            num_kv_shared_layers,
        ),
    )
    .expect("config");
    let layer_count = num_hidden_layers as usize;
    let mut per_layer_embedding = vec![0.0; 3 * layer_count];
    if per_layer_inputs {
        per_layer_embedding[0] = 1.0;
    }
    let mut tensors = vec![
        (
            "model.language_model.embed_tokens.weight",
            vec![3, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        ),
        ("model.language_model.norm.weight", vec![2], vec![1.0, 1.0]),
        (
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.pre_feedforward_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.post_feedforward_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.layer_scalar",
            vec![1],
            vec![1.0],
        ),
    ];
    if per_layer_inputs {
        tensors.push((
            "model.language_model.embed_tokens_per_layer.weight",
            vec![3, layer_count],
            per_layer_embedding,
        ));
        tensors.push((
            "model.language_model.per_layer_model_projection.weight",
            vec![layer_count, 2],
            vec![0.0; 2 * layer_count],
        ));
        tensors.push((
            "model.language_model.per_layer_projection_norm.weight",
            vec![1],
            vec![1.0],
        ));
        tensors.push((
            "model.language_model.layers.0.per_layer_input_gate.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ));
        tensors.push((
            "model.language_model.layers.0.per_layer_projection.weight",
            vec![2, 1],
            vec![1.0, 0.0],
        ));
        tensors.push((
            "model.language_model.layers.0.post_per_layer_input_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ));
    }
    if num_hidden_layers > 1 {
        tensors.extend([
            (
                "model.language_model.layers.1.input_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.self_attn.q_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.self_attn.q_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.self_attn.o_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.post_attention_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.pre_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.mlp.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.mlp.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.mlp.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.post_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.layer_scalar",
                vec![1],
                vec![1.0],
            ),
        ]);
        if per_layer_inputs {
            tensors.extend([
                (
                    "model.language_model.layers.1.per_layer_input_gate.weight",
                    vec![1, 2],
                    vec![0.0, 0.0],
                ),
                (
                    "model.language_model.layers.1.per_layer_projection.weight",
                    vec![2, 1],
                    vec![0.0, 0.0],
                ),
                (
                    "model.language_model.layers.1.post_per_layer_input_norm.weight",
                    vec![2],
                    vec![1.0, 1.0],
                ),
            ]);
        }
    }
    let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
    std::fs::write(root.join("model.safetensors"), &safetensors).expect("safetensors");
    let weight_map = tensors
        .iter()
        .map(|(tensor, _, _)| {
            (
                (*tensor).to_owned(),
                serde_json::Value::String("model.safetensors".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        json!({
            "metadata": {"total_size": safetensors.len()},
            "weight_map": weight_map
        })
        .to_string(),
    )
    .expect("index");
}

fn tiny_gemma4_config(
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    layer_types: &[&str],
) -> String {
    tiny_gemma4_config_with_options(
        num_hidden_layers,
        max_position_embeddings,
        layer_types,
        false,
        0,
    )
}

fn tiny_gemma4_config_with_options(
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    layer_types: &[&str],
    per_layer_inputs: bool,
    num_kv_shared_layers: u32,
) -> String {
    json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": false,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": false,
            "global_head_dim": null,
            "head_dim": 2,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2,
            "hidden_size_per_layer_input": if per_layer_inputs { 1 } else { 0 },
            "intermediate_size": 1,
            "layer_types": layer_types,
            "max_position_embeddings": max_position_embeddings,
            "model_type": "gemma4_text",
            "num_attention_heads": 1,
            "num_global_key_value_heads": null,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": num_kv_shared_layers,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "sliding_window": 2,
            "tie_word_embeddings": true,
            "use_double_wide_mlp": false,
            "vocab_size": 3,
            "vocab_size_per_layer_input": 3
        },
        "tie_word_embeddings": true
    })
    .to_string()
}

fn tiny_owned_multi_safetensors_bf16(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let end = data.len();
        header.insert(
            (*name).to_owned(),
            json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    bytes
}

fn temp_snapshot_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("kir-ai-{name}-{nanos}"))
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {idx}: expected {expected}, got {actual}"
        );
    }
}
