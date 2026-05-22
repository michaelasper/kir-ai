use llm_models::SafetensorsIndex;

fn index_json(total_size: &str) -> String {
    format!(
        r#"{{
            "metadata": {{ "total_size": {total_size} }},
            "weight_map": {{ "tensor.weight": "model.safetensors" }}
        }}"#
    )
}

#[test]
fn preserves_total_size_integer_larger_than_f64_exact_range() {
    let index = SafetensorsIndex::from_json(&index_json("9007199254740993"))
        .expect("integer total_size above f64 exact range should parse");

    assert_eq!(index.total_size_bytes, 9_007_199_254_740_993);
}

#[test]
fn preserves_float_encoded_total_size_at_f64_exact_integer_limit() {
    let index = SafetensorsIndex::from_json(&index_json("9007199254740992.0"))
        .expect("float-encoded exact integer total_size should parse");

    assert_eq!(index.total_size_bytes, 9_007_199_254_740_992);
}

#[test]
fn rejects_float_encoded_total_size_outside_exact_integer_range() {
    let err = SafetensorsIndex::from_json(&index_json("9007199254740993.0"))
        .expect_err("float-encoded imprecise total_size must fail closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.to_string().contains("total_size"));
}

#[test]
fn rejects_fractional_total_size() {
    let err = SafetensorsIndex::from_json(&index_json("12.5"))
        .expect_err("fractional total_size must fail closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.to_string().contains("total_size"));
}

#[test]
fn rejects_fractional_total_size_near_f64_precision_boundary() {
    for total_size in ["9007199254740992.1", "9007199254740991.5"] {
        let err = SafetensorsIndex::from_json(&index_json(total_size))
            .expect_err("fractional total_size must fail closed before f64 rounding");

        assert_eq!(err.code(), "invalid_request");
        assert!(err.to_string().contains("total_size"));
    }
}

#[test]
fn rejects_total_size_larger_than_u64() {
    let err = SafetensorsIndex::from_json(&index_json("18446744073709551616"))
        .expect_err("out-of-range total_size must fail closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.to_string().contains("total_size"));
}
