use super::admin::*;
use schemars::schema_for;
use std::fs;

#[test]
fn generate_admin_api_schemas() {
    let schemas = [
        ("HealthResponse", schema_for!(HealthResponse)),
        ("AdminModelListResponse", schema_for!(AdminModelListResponse)),
        ("AdminModelStatusResponse", schema_for!(AdminModelStatusResponse)),
        ("AdminModelVerifyResponse", schema_for!(AdminModelVerifyResponse)),
        ("AdminModelPullResponse", schema_for!(AdminModelPullResponse)),
        ("AdminMetricsResponse", schema_for!(AdminMetricsResponse)),
    ];

    let output_dir = "../../docs/schemas/admin";
    fs::create_dir_all(output_dir).expect("failed to create schema directory");

    for (name, schema) in schemas {
        let json = serde_json::to_string_pretty(&schema).expect("failed to serialize schema");
        fs::write(format!("{}/{}.json", output_dir, name), json).expect("failed to write schema");
    }
}
