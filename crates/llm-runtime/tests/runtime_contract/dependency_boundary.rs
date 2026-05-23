use serde_json::Value;
use std::{path::Path, process::Command};

#[test]
fn runtime_depends_on_backend_contracts_without_normal_native_backend_dependency() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("llm-runtime lives under crates in the workspace");
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata runs for workspace");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata: Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
    let packages = metadata["packages"]
        .as_array()
        .expect("metadata packages is an array");
    let runtime = packages
        .iter()
        .find(|package| package["name"].as_str() == Some("llm-runtime"))
        .expect("llm-runtime package is present");
    let dependencies = runtime["dependencies"]
        .as_array()
        .expect("runtime dependencies are listed");

    let normal_dependencies = dependencies
        .iter()
        .filter(|dependency| dependency["kind"].is_null())
        .collect::<Vec<_>>();

    assert!(
        normal_dependencies
            .iter()
            .any(
                |dependency| dependency["name"].as_str() == Some("llm-backend-contracts")
                    && !dependency["optional"].as_bool().unwrap_or(false)
            ),
        "llm-runtime must have a non-optional dependency on llm-backend-contracts"
    );
    assert!(
        normal_dependencies
            .iter()
            .all(
                |dependency| dependency["name"].as_str() != Some("llm-backend")
                    || dependency["optional"].as_bool() == Some(true)
            ),
        "llm-runtime must not require the native llm-backend crate for production code"
    );
}
