use std::path::PathBuf;
use std::process::Command;

#[test]
fn runtime_prompt_rendering_does_not_depend_on_hf_tokenizer_stack() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("llm-runtime lives under crates/ in the workspace");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .current_dir(workspace)
        .args([
            "tree",
            "-p",
            "llm-runtime",
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("cargo tree runs for llm-runtime");

    assert!(
        output.status.success(),
        "cargo tree failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let tree = String::from_utf8(output.stdout).expect("cargo tree emits utf-8");
    for forbidden in ["llm-tokenizer", "tokenizers", "onig"] {
        assert!(
            !tree.lines().any(|line| line.starts_with(forbidden)),
            "llm-runtime normal dependency graph still includes `{forbidden}`:\n{tree}"
        );
    }
}
