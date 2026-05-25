use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SHADER_FILES: &[&str] = &[
    "src/device/shaders/generic.metal",
    "src/device/shaders/transformer.metal",
    "src/device/shaders/matvec.metal",
    "src/device/shaders/reductions.metal",
];

fn main() {
    if let Err(err) = generate_shader_artifact() {
        println!("cargo:warning=llm-metal shader artifact generation failed: {err}");
        std::process::exit(1);
    }
}

fn generate_shader_artifact() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/device/shaders.rs");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");

    let manifest_dir = PathBuf::from(required_env("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(required_env("OUT_DIR")?);
    let source = read_shader_source(&manifest_dir)?;
    let source_hash = sha256_hex(source.as_bytes());
    let metallib = compile_metallib(&out_dir, &source, &source_hash);
    let generated = generated_shader_module(&source_hash, metallib.as_deref());

    fs::write(out_dir.join("shader_metallib.rs"), generated)?;
    Ok(())
}

fn required_env(name: &str) -> Result<OsString, Box<dyn Error>> {
    env::var_os(name).ok_or_else(|| format!("{name} is not set").into())
}

fn read_shader_source(manifest_dir: &Path) -> Result<String, Box<dyn Error>> {
    let mut source = String::new();
    for shader_file in SHADER_FILES {
        println!("cargo:rerun-if-changed={shader_file}");
        source.push_str(&fs::read_to_string(manifest_dir.join(shader_file))?);
    }
    Ok(source)
}

fn compile_metallib(out_dir: &Path, source: &str, source_hash: &str) -> Option<PathBuf> {
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return None;
    }

    let source_path = out_dir.join(format!("kir_ai_shaders-{source_hash}.metal"));
    let air_path = out_dir.join(format!("kir_ai_shaders-{source_hash}.air"));
    let metallib_path = out_dir.join(format!("kir_ai_shaders-{source_hash}.metallib"));

    if fs::write(&source_path, source).is_err() {
        return None;
    }

    let metal_output = Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-c")
        .arg(&source_path)
        .arg("-o")
        .arg(&air_path)
        .output();
    if !command_succeeded(metal_output) {
        return None;
    }

    let metallib_output = Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("metallib")
        .arg(&air_path)
        .arg("-o")
        .arg(&metallib_path)
        .output();
    if !command_succeeded(metallib_output) {
        return None;
    }

    metallib_path.exists().then_some(metallib_path)
}

fn command_succeeded(output: std::io::Result<Output>) -> bool {
    output.is_ok_and(|output| output.status.success())
}

fn generated_shader_module(source_hash: &str, metallib: Option<&Path>) -> String {
    let metallib_const = match metallib {
        Some(path) => {
            let path_string = path.to_string_lossy();
            let path_literal = rust_string_literal(path_string.as_ref());
            format!(
                "const EMBEDDED_METALLIB_BYTES: &[u8] = include_bytes!({path_literal});\n\
                 pub(crate) const EMBEDDED_METALLIB: Option<&'static [u8]> = Some(EMBEDDED_METALLIB_BYTES);\n"
            )
        }
        None => "pub(crate) const EMBEDDED_METALLIB: Option<&'static [u8]> = None;\n".to_owned(),
    };

    format!(
        "pub(crate) const SHADER_SOURCE_SHA256: &str = {};\n{metallib_const}",
        rust_string_literal(source_hash)
    )
}

fn rust_string_literal(value: &str) -> String {
    let mut literal = String::with_capacity(value.len() + 2);
    literal.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => literal.push_str("\\\\"),
            '"' => literal.push_str("\\\""),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            '\t' => literal.push_str("\\t"),
            _ => literal.push(ch),
        }
    }
    literal.push('"');
    literal
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(hex_digit(byte >> 4));
        encoded.push(hex_digit(byte & 0x0f));
    }
    encoded
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        10..=15 => char::from(b'a' + value - 10),
        _ => '?',
    }
}
