use std::path::PathBuf;
use std::process::Command;

#[test]
fn engine_default_dependency_graph_does_not_include_llm_metal() {
    let tree = cargo_tree(&[]);

    assert_no_llm_metal(&tree);
}

#[test]
fn engine_native_text_dependency_graph_does_not_include_llm_metal_without_metal_feature() {
    let tree = cargo_tree(&[
        "--no-default-features",
        "--features",
        "native-qwen,native-gemma",
    ]);

    assert_no_llm_metal(&tree);
}

#[test]
fn engine_metal_feature_enables_llm_metal_dependency() {
    let tree = cargo_tree(&["--no-default-features", "--features", "native-qwen,metal"]);

    assert!(
        has_llm_metal(&tree),
        "llm-engine `metal` feature should enable `llm-metal`:\n{tree}"
    );
}

fn cargo_tree(feature_args: &[&str]) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("llm-engine lives under crates/ in the workspace");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .current_dir(workspace)
        .args([
            "tree",
            "-p",
            "llm-engine",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .args(feature_args)
        .output()
        .expect("cargo tree runs for llm-engine");

    assert!(
        output.status.success(),
        "cargo tree failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).expect("cargo tree emits utf-8")
}

fn assert_no_llm_metal(tree: &str) {
    assert!(
        !has_llm_metal(tree),
        "llm-engine normal dependency graph unexpectedly includes `llm-metal`:\n{tree}"
    );
}

fn has_llm_metal(tree: &str) -> bool {
    tree.lines().any(|line| line.starts_with("llm-metal "))
}
