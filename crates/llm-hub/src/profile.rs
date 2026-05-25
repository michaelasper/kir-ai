use llm_models::{BackendKind, ModelFamily};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const DEFAULT_MODEL_PROFILE_NAME: &str = "qwen36-safetensors-bf16";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelProfile {
    pub name: String,
    #[serde(with = "model_family_slug")]
    #[schemars(with = "String")]
    pub family: ModelFamily,
    #[serde(with = "backend_kind_slug")]
    #[schemars(with = "String")]
    pub loader: BackendKind,
    pub quantization: String,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileArtifactSet {
    Qwen,
    Gemma,
    Llama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BuiltinProfile {
    name: &'static str,
    family: ModelFamily,
    loader: BackendKind,
    quantization: &'static str,
}

const BUILTIN_PROFILES: &[BuiltinProfile] = &[
    BuiltinProfile {
        name: "gemma4-e2b-it-mlx-4bit",
        family: ModelFamily::Gemma,
        loader: BackendKind::Mlx,
        quantization: "4bit",
    },
    BuiltinProfile {
        name: "gemma4-text-safetensors-bf16",
        family: ModelFamily::Gemma,
        loader: BackendKind::NativeMetal,
        quantization: "bf16",
    },
    BuiltinProfile {
        name: "llama32-3b-instruct-mlx-4bit",
        family: ModelFamily::Llama,
        loader: BackendKind::Mlx,
        quantization: "4bit",
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-4bit",
        family: ModelFamily::Qwen,
        loader: BackendKind::Mlx,
        quantization: "4bit",
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-8bit",
        family: ModelFamily::Qwen,
        loader: BackendKind::Mlx,
        quantization: "8bit",
    },
    BuiltinProfile {
        name: "qwen35-4b-mlx-optiq-4bit",
        family: ModelFamily::Qwen,
        loader: BackendKind::Mlx,
        quantization: "optiq-4bit",
    },
    BuiltinProfile {
        name: "qwen3-dense-safetensors-bf16",
        family: ModelFamily::Qwen,
        loader: BackendKind::NativeMetal,
        quantization: "bf16",
    },
    BuiltinProfile {
        name: "qwen36-mlx-4bit",
        family: ModelFamily::Qwen,
        loader: BackendKind::Mlx,
        quantization: "4bit",
    },
    BuiltinProfile {
        name: "qwen36-safetensors-bf16",
        family: ModelFamily::Qwen,
        loader: BackendKind::NativeMetal,
        quantization: "bf16",
    },
];

impl ModelProfile {
    pub fn builtin(name: &str) -> Option<Self> {
        BUILTIN_PROFILES
            .iter()
            .copied()
            .find(|profile| profile.name == name)
            .and_then(Self::from_builtin)
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
        Self::builtin(DEFAULT_MODEL_PROFILE_NAME).expect("built-in qwen36 safetensors BF16 profile")
    }

    pub fn qwen3_dense_safetensors_bf16() -> Self {
        Self::builtin("qwen3-dense-safetensors-bf16")
            .expect("built-in qwen3 dense safetensors BF16 profile")
    }

    pub fn gemma4_text_safetensors_bf16() -> Self {
        Self::builtin("gemma4-text-safetensors-bf16")
            .expect("built-in Gemma 4 text safetensors BF16 profile")
    }

    pub fn gemma4_e2b_it_mlx_4bit() -> Self {
        Self::builtin("gemma4-e2b-it-mlx-4bit").expect("built-in Gemma 4 E2B IT MLX 4-bit profile")
    }

    pub fn llama32_3b_instruct_mlx_4bit() -> Self {
        Self::builtin("llama32-3b-instruct-mlx-4bit")
            .expect("built-in Llama 3.2 3B Instruct MLX 4-bit profile")
    }

    fn from_builtin(profile: BuiltinProfile) -> Option<Self> {
        let (allow_patterns, ignore_patterns) = match artifact_set_for_family(profile.family)? {
            ProfileArtifactSet::Qwen => (
                qwen_static_and_safetensors_patterns(),
                qwen_ignore_patterns(),
            ),
            ProfileArtifactSet::Gemma => (
                gemma_text_static_and_safetensors_patterns(),
                gemma_text_ignore_patterns(),
            ),
            ProfileArtifactSet::Llama => (
                llama_text_static_and_safetensors_patterns(),
                llama_text_ignore_patterns(),
            ),
        };
        Some(Self {
            name: profile.name.to_owned(),
            family: profile.family,
            loader: profile.loader,
            quantization: profile.quantization.to_owned(),
            allow_patterns,
            ignore_patterns,
        })
    }

    pub fn family_slug(&self) -> &'static str {
        self.family.canonical_slug()
    }

    pub fn loader_slug(&self) -> &'static str {
        self.loader.canonical_slug()
    }
}

fn artifact_set_for_family(family: ModelFamily) -> Option<ProfileArtifactSet> {
    match family {
        ModelFamily::Qwen | ModelFamily::DeepSeek => Some(ProfileArtifactSet::Qwen),
        ModelFamily::Gemma => Some(ProfileArtifactSet::Gemma),
        ModelFamily::Llama => Some(ProfileArtifactSet::Llama),
        _ => None,
    }
}

mod model_family_slug {
    use super::*;

    pub fn serialize<S>(family: &ModelFamily, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(family.canonical_slug())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ModelFamily, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        ModelFamily::parse_slug(&value).map_err(serde::de::Error::custom)
    }
}

mod backend_kind_slug {
    use super::*;

    pub fn serialize<S>(backend: &BackendKind, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(backend.canonical_slug())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BackendKind, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        BackendKind::parse_slug(&value).map_err(serde::de::Error::custom)
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
        "*.jinja".to_owned(),
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

fn llama_text_static_and_safetensors_patterns() -> Vec<String> {
    vec![
        "*.json".to_owned(),
        "*.jinja".to_owned(),
        "*.model".to_owned(),
        "*.txt".to_owned(),
        "tokenizer*".to_owned(),
        "README.md".to_owned(),
        "LICENSE*".to_owned(),
        "*.safetensors".to_owned(),
        "*.safetensors.index.json".to_owned(),
    ]
}

fn llama_text_ignore_patterns() -> Vec<String> {
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
