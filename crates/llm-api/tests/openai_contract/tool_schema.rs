use super::*;

#[test]
fn canonical_tool_schema_json_matches_equivalent_property_and_required_order() {
    let current = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "required": ["query", "source"],
            "properties": {
                "query": {"type": "string"},
                "source": {"type": "string"}
            },
            "additionalProperties": false
        }),
    )];
    let permuted = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "additionalProperties": false,
            "properties": {
                "source": {"type": "string"},
                "query": {"type": "string"}
            },
            "required": ["source", "query"],
            "type": "object"
        }),
    )];

    assert_ne!(
        serde_json::to_string(&current).expect("current serializes"),
        serde_json::to_string(&permuted).expect("permuted serializes")
    );
    assert_eq!(
        canonical_tool_schema_json(&current).expect("current canonicalizes"),
        canonical_tool_schema_json(&permuted).expect("permuted canonicalizes")
    );

    let canonical = canonicalize_tool_schemas(&permuted);
    assert_eq!(
        canonical[0].function.parameters["required"],
        json!(["query", "source"])
    );
    assert_eq!(
        canonical[0].function.parameters["properties"]
            .as_object()
            .expect("properties object")
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        ["query", "source"]
    );
}

#[test]
fn canonical_tool_schema_preserves_tool_order_names_and_descriptions() {
    let tools = vec![
        ToolDefinition::function("second", "Second tool.", json!({"type": "object"})),
        ToolDefinition::function("first", "First tool.", json!({"type": "object"})),
    ];

    let canonical = canonicalize_tool_schemas(&tools);

    assert_eq!(canonical[0].function.name, "second");
    assert_eq!(
        canonical[0].function.description.as_deref(),
        Some("Second tool.")
    );
    assert_eq!(canonical[1].function.name, "first");
    assert_eq!(
        canonical[1].function.description.as_deref(),
        Some("First tool.")
    );
}

#[test]
fn canonical_tool_schema_keeps_semantic_differences_distinct() {
    let alpha_then_beta = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["alpha", "beta"]}
            }
        }),
    )];
    let beta_then_alpha = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"enum": ["beta", "alpha"], "type": "string"}
            }
        }),
    )];
    let different_description = vec![ToolDefinition::function(
        "lookup",
        "Lookup other docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["alpha", "beta"]}
            }
        }),
    )];

    assert_ne!(
        canonical_tool_schema_json(&alpha_then_beta).expect("canonical alpha/beta"),
        canonical_tool_schema_json(&beta_then_alpha).expect("canonical beta/alpha")
    );
    assert_ne!(
        canonical_tool_schema_json(&alpha_then_beta).expect("canonical alpha/beta"),
        canonical_tool_schema_json(&different_description).expect("canonical description")
    );
}
