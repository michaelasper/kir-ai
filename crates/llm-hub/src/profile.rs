use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub family: String,
    pub loader: String,
    pub quantization: String,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

impl ModelProfile {
    pub fn builtin(name: &str) -> Option<Self> {
        match name {
            "gemma4-text-safetensors-bf16" => Some(Self::gemma4_text_safetensors_bf16()),
            "qwen35-4b-mlx-4bit" => Some(Self::qwen35_4b_mlx_4bit()),
            "qwen3-dense-safetensors-bf16" => Some(Self::qwen3_dense_safetensors_bf16()),
            "qwen36-mlx-4bit" => Some(Self::qwen36_mlx_4bit()),
            "qwen36-safetensors-bf16" => Some(Self::qwen36_safetensors_bf16()),
            _ => None,
        }
    }

    pub fn qwen35_4b_mlx_4bit() -> Self {
        Self::qwen_mlx_4bit("qwen35-4b-mlx-4bit")
    }

    pub fn qwen36_mlx_4bit() -> Self {
        Self::qwen_mlx_4bit("qwen36-mlx-4bit")
    }

    pub fn qwen36_safetensors_bf16() -> Self {
        Self {
            name: "qwen36-safetensors-bf16".to_owned(),
            family: "qwen".to_owned(),
            loader: "native-metal".to_owned(),
            quantization: "bf16".to_owned(),
            allow_patterns: qwen_static_and_safetensors_patterns(),
            ignore_patterns: qwen_ignore_patterns(),
        }
    }

    pub fn qwen3_dense_safetensors_bf16() -> Self {
        Self {
            name: "qwen3-dense-safetensors-bf16".to_owned(),
            family: "qwen".to_owned(),
            loader: "native-metal".to_owned(),
            quantization: "bf16".to_owned(),
            allow_patterns: qwen_static_and_safetensors_patterns(),
            ignore_patterns: qwen_ignore_patterns(),
        }
    }

    pub fn gemma4_text_safetensors_bf16() -> Self {
        Self {
            name: "gemma4-text-safetensors-bf16".to_owned(),
            family: "gemma".to_owned(),
            loader: "mlx".to_owned(),
            quantization: "bf16".to_owned(),
            allow_patterns: gemma_text_static_and_safetensors_patterns(),
            ignore_patterns: gemma_text_ignore_patterns(),
        }
    }

    fn qwen_mlx_4bit(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            family: "qwen".to_owned(),
            loader: "mlx".to_owned(),
            quantization: "4bit".to_owned(),
            allow_patterns: qwen_static_and_safetensors_patterns(),
            ignore_patterns: qwen_ignore_patterns(),
        }
    }
}

fn qwen_static_and_safetensors_patterns() -> Vec<String> {
    vec![
        "*.json".to_owned(),
        "*.jinja".to_owned(),
        "*.txt".to_owned(),
        "tokenizer*".to_owned(),
        "README.md".to_owned(),
        "LICENSE*".to_owned(),
        "*.safetensors".to_owned(),
        "*.safetensors.index.json".to_owned(),
    ]
}

fn qwen_ignore_patterns() -> Vec<String> {
    vec![
        "*.bin".to_owned(),
        "*.pt".to_owned(),
        "optimizer*".to_owned(),
        "training_args.bin".to_owned(),
    ]
}

fn gemma_text_static_and_safetensors_patterns() -> Vec<String> {
    vec![
        "*.json".to_owned(),
        "*.model".to_owned(),
        "*.txt".to_owned(),
        "tokenizer*".to_owned(),
        "README.md".to_owned(),
        "LICENSE*".to_owned(),
        "*.safetensors".to_owned(),
        "*.safetensors.index.json".to_owned(),
    ]
}

fn gemma_text_ignore_patterns() -> Vec<String> {
    vec![
        "*.bin".to_owned(),
        "*.pt".to_owned(),
        "optimizer*".to_owned(),
        "training_args.bin".to_owned(),
        "image_*".to_owned(),
        "preprocessor_config.json".to_owned(),
        "processor_config.json".to_owned(),
        "vision*".to_owned(),
        "mm_projector*".to_owned(),
        "multi_modal_projector*".to_owned(),
        "projector*".to_owned(),
    ]
}
