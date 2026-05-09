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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileArtifactSet {
    QwenText,
    GemmaText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BuiltinProfile {
    name: &'static str,
    family: &'static str,
    loader: &'static str,
    quantization: &'static str,
    artifacts: ProfileArtifactSet,
}

const BUILTIN_PROFILES: &[BuiltinProfile] = &[
    BuiltinProfile {
        name: "gemma4-text-safetensors-bf16",
        family: "gemma",
        loader: "mlx",
        quantization: "bf16",
        artifacts: ProfileArtifactSet::GemmaText,
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-4bit",
        family: "qwen",
        loader: "mlx",
        quantization: "4bit",
        artifacts: ProfileArtifactSet::QwenText,
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-8bit",
        family: "qwen",
        loader: "mlx",
        quantization: "8bit",
        artifacts: ProfileArtifactSet::QwenText,
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-optiq-4bit",
        family: "qwen",
        loader: "mlx",
        quantization: "optiq-4bit",
        artifacts: ProfileArtifactSet::QwenText,
    },
    BuiltinProfile {
        name: "qwen3-dense-safetensors-bf16",
        family: "qwen",
        loader: "native-metal",
        quantization: "bf16",
        artifacts: ProfileArtifactSet::QwenText,
    },
    BuiltinProfile {
        name: "qwen36-mlx-4bit",
        family: "qwen",
        loader: "mlx",
        quantization: "4bit",
        artifacts: ProfileArtifactSet::QwenText,
    },
    BuiltinProfile {
        name: "qwen36-safetensors-bf16",
        family: "qwen",
        loader: "native-metal",
        quantization: "bf16",
        artifacts: ProfileArtifactSet::QwenText,
    },
];

impl ModelProfile {
    pub fn builtin(name: &str) -> Option<Self> {
        BUILTIN_PROFILES
            .iter()
            .copied()
            .find(|profile| profile.name == name)
            .map(Self::from_builtin)
    }

    pub fn builtin_names() -> impl Iterator<Item = &'static str> {
        BUILTIN_PROFILES.iter().map(|profile| profile.name)
    }

    pub fn qwen35_4b_mlx_4bit() -> Self {
        Self::builtin("qwen35-4b-mlx-4bit").expect("built-in qwen35 4B MLX 4-bit profile")
    }

    pub fn qwen35_4b_mlx_8bit() -> Self {
        Self::builtin("qwen35-4b-mlx-8bit").expect("built-in qwen35 4B MLX 8-bit profile")
    }

    pub fn qwen35_4b_mlx_optiq_4bit() -> Self {
        Self::builtin("qwen35-4b-mlx-optiq-4bit")
            .expect("built-in qwen35 4B MLX OptiQ 4-bit profile")
    }

    pub fn qwen36_mlx_4bit() -> Self {
        Self::builtin("qwen36-mlx-4bit").expect("built-in qwen36 MLX 4-bit profile")
    }

    pub fn qwen36_safetensors_bf16() -> Self {
        Self::builtin("qwen36-safetensors-bf16").expect("built-in qwen36 safetensors BF16 profile")
    }

    pub fn qwen3_dense_safetensors_bf16() -> Self {
        Self::builtin("qwen3-dense-safetensors-bf16")
            .expect("built-in qwen3 dense safetensors BF16 profile")
    }

    pub fn gemma4_text_safetensors_bf16() -> Self {
        Self::builtin("gemma4-text-safetensors-bf16")
            .expect("built-in Gemma 4 text safetensors BF16 profile")
    }

    fn from_builtin(profile: BuiltinProfile) -> Self {
        let (allow_patterns, ignore_patterns) = match profile.artifacts {
            ProfileArtifactSet::QwenText => (
                qwen_static_and_safetensors_patterns(),
                qwen_ignore_patterns(),
            ),
            ProfileArtifactSet::GemmaText => (
                gemma_text_static_and_safetensors_patterns(),
                gemma_text_ignore_patterns(),
            ),
        };
        Self {
            name: profile.name.to_owned(),
            family: profile.family.to_owned(),
            loader: profile.loader.to_owned(),
            quantization: profile.quantization.to_owned(),
            allow_patterns,
            ignore_patterns,
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
        "image_*".to_owned(),
        "preprocessor_config.json".to_owned(),
        "processor_config.json".to_owned(),
        "optimizer*".to_owned(),
        "training_args.bin".to_owned(),
        "video_preprocessor_config.json".to_owned(),
        "vision*".to_owned(),
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
        "video_preprocessor_config.json".to_owned(),
        "vision*".to_owned(),
        "mm_projector*".to_owned(),
        "multi_modal_projector*".to_owned(),
        "projector*".to_owned(),
    ]
}
