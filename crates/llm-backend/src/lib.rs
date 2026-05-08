use async_trait::async_trait;
use llm_api::FinishReason;
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    ops::Range,
    path::{Path, PathBuf},
};
use thiserror::Error;

const MAX_SAFETENSORS_HEADER_LEN: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: FinishReason,
}

#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError>;
}

#[derive(Debug, Clone)]
pub struct DeterministicBackend {
    model_id: String,
    text: String,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            text: text.into(),
        }
    }
}

#[async_trait]
impl ModelBackend for DeterministicBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        Ok(BackendOutput {
            text: self.text.clone(),
            prompt_tokens: count_tokens(&request.prompt),
            completion_tokens: count_tokens(&self.text),
            finish_reason: FinishReason::Stop,
        })
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelNotFound {
        requested: String,
        available: String,
    },
    #[error("backend error: {0}")]
    Other(String),
}

fn count_tokens(text: &str) -> u64 {
    let normalized = text
        .replace("<|im_start|>system", " ")
        .replace("<|im_start|>user", " ")
        .replace("<|im_start|>assistant", " ")
        .replace("<|im_start|>tool", " ")
        .replace("<|im_end|>", " ")
        .replace("<think>", " ")
        .replace("</think>", " ");
    normalized.split_whitespace().count().max(1) as u64
}

#[derive(Debug, Clone)]
pub struct SafeTensorArchive {
    bytes: Vec<u8>,
}

impl SafeTensorArchive {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorLoadError> {
        SafeTensors::deserialize(bytes)
            .map_err(|err| TensorLoadError::integrity(format!("invalid safetensors: {err}")))?;
        Ok(Self {
            bytes: bytes.to_vec(),
        })
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        let tensors = self.tensors()?;
        let view = tensors
            .tensor(name)
            .map_err(|err| TensorLoadError::missing(format!("tensor `{name}` not found: {err}")))?;
        Ok(TensorMetadata {
            name: name.to_owned(),
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            byte_len: view.data().len(),
        })
    }

    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, TensorLoadError> {
        let tensors = self.tensors()?;
        let view = tensors
            .tensor(name)
            .map_err(|err| TensorLoadError::missing(format!("tensor `{name}` not found: {err}")))?;
        if view.dtype() != Dtype::F32 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` has dtype {:?}, expected F32",
                view.dtype()
            )));
        }
        let data = view.data();
        if data.len() % std::mem::size_of::<f32>() != 0 {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` byte length is not divisible by 4"
            )));
        }
        Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk has length 4")))
            .collect())
    }

    fn tensors(&self) -> Result<SafeTensors<'_>, TensorLoadError> {
        SafeTensors::deserialize(&self.bytes)
            .map_err(|err| TensorLoadError::integrity(format!("invalid safetensors: {err}")))
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorHeader {
    source_path: Option<PathBuf>,
    file_len: u64,
    header_len: u64,
    data_start: u64,
    tensors: BTreeMap<String, TensorHeaderEntry>,
}

impl SafeTensorHeader {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorLoadError> {
        let file_len = bytes.len() as u64;
        let (header_len, header_end) = read_header_prefix(bytes, file_len)?;
        let header = bytes
            .get(8..header_end)
            .ok_or_else(|| TensorLoadError::integrity("safetensors header is truncated"))?;
        Self::from_header_bytes(None, file_len, header_len, header)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not open safetensors file `{}`: {err}",
                path.display()
            ))
        })?;
        let file_len = file
            .metadata()
            .map_err(|err| {
                TensorLoadError::integrity(format!(
                    "could not read metadata for `{}`: {err}",
                    path.display()
                ))
            })?
            .len();
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix).map_err(|err| {
            TensorLoadError::integrity(format!(
                "could not read safetensors header prefix from `{}`: {err}",
                path.display()
            ))
        })?;
        let header_len = validate_header_len(u64::from_le_bytes(prefix), file_len)?;
        let mut header = vec![0_u8; usize_from_u64(header_len, "safetensors header is too large")?];
        file.read_exact(&mut header).map_err(|err| {
            TensorLoadError::integrity(format!(
                "could not read safetensors header from `{}`: {err}",
                path.display()
            ))
        })?;
        Self::from_header_bytes(Some(path.to_path_buf()), file_len, header_len, &header)
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn header_len(&self) -> u64 {
        self.header_len
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        let tensor = self.tensor_entry(name)?;
        Ok(TensorMetadata {
            name: name.to_owned(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            byte_len: tensor.byte_len()?,
        })
    }

    pub fn tensor_data_range(&self, name: &str) -> Result<Range<u64>, TensorLoadError> {
        let tensor = self.tensor_entry(name)?;
        let start = self
            .data_start
            .checked_add(tensor.data_offsets[0])
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        let end = self
            .data_start
            .checked_add(tensor.data_offsets[1])
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        Ok(start..end)
    }

    fn from_header_bytes(
        source_path: Option<PathBuf>,
        file_len: u64,
        header_len: u64,
        header: &[u8],
    ) -> Result<Self, TensorLoadError> {
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
        let payload_len = file_len
            .checked_sub(data_start)
            .ok_or_else(|| TensorLoadError::integrity("safetensors payload is truncated"))?;
        let root: serde_json::Value = serde_json::from_slice(header).map_err(|err| {
            TensorLoadError::integrity(format!("invalid safetensors header json: {err}"))
        })?;
        let object = root.as_object().ok_or_else(|| {
            TensorLoadError::integrity("safetensors header must be a json object")
        })?;
        let mut tensors = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            tensors.insert(
                name.clone(),
                TensorHeaderEntry::from_json(name, value, payload_len)?,
            );
        }
        if tensors.is_empty() {
            return Err(TensorLoadError::integrity(
                "safetensors header does not contain tensors",
            ));
        }
        Ok(Self {
            source_path,
            file_len,
            header_len,
            data_start,
            tensors,
        })
    }

    fn tensor_entry(&self, name: &str) -> Result<&TensorHeaderEntry, TensorLoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| TensorLoadError::missing(format!("tensor `{name}` not found")))
    }
}

#[derive(Debug, Clone)]
struct TensorHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl TensorHeaderEntry {
    fn from_json(
        name: &str,
        value: &serde_json::Value,
        payload_len: u64,
    ) -> Result<Self, TensorLoadError> {
        let object = value.as_object().ok_or_else(|| {
            TensorLoadError::integrity(format!("tensor `{name}` header must be an object"))
        })?;
        let dtype = object
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` is missing dtype")))?
            .to_owned();
        let shape = parse_shape(name, object.get("shape"))?;
        let data_offsets = parse_data_offsets(name, object.get("data_offsets"))?;
        if data_offsets[1] < data_offsets[0] {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` has inverted data offsets"
            )));
        }
        if data_offsets[1] > payload_len {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` data offsets exceed payload length"
            )));
        }
        Ok(Self {
            dtype,
            shape,
            data_offsets,
        })
    }

    fn byte_len(&self) -> Result<usize, TensorLoadError> {
        usize_from_u64(
            self.data_offsets[1] - self.data_offsets[0],
            "tensor byte length does not fit in usize",
        )
    }
}

fn read_header_prefix(bytes: &[u8], file_len: u64) -> Result<(u64, usize), TensorLoadError> {
    let prefix = bytes
        .get(0..8)
        .ok_or_else(|| TensorLoadError::integrity("safetensors file is missing header prefix"))?;
    let header_len = validate_header_len(
        u64::from_le_bytes(prefix.try_into().expect("prefix has length 8")),
        file_len,
    )?;
    let header_end = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
    Ok((
        header_len,
        usize_from_u64(header_end, "safetensors header end does not fit in usize")?,
    ))
}

fn validate_header_len(header_len: u64, file_len: u64) -> Result<u64, TensorLoadError> {
    if header_len > MAX_SAFETENSORS_HEADER_LEN {
        return Err(TensorLoadError::integrity(format!(
            "safetensors header length {header_len} exceeds limit {MAX_SAFETENSORS_HEADER_LEN}"
        )));
    }
    let header_end = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
    if header_end > file_len {
        return Err(TensorLoadError::integrity(
            "safetensors header length exceeds file length",
        ));
    }
    Ok(header_len)
}

fn parse_shape(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<Vec<usize>, TensorLoadError> {
    let array = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` is missing shape")))?;
    array
        .iter()
        .map(|value| {
            let dim = value.as_u64().ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` shape must contain integers"))
            })?;
            usize_from_u64(dim, "tensor shape dimension does not fit in usize")
        })
        .collect()
}

fn parse_data_offsets(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<[u64; 2], TensorLoadError> {
    let array = value.and_then(serde_json::Value::as_array).ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` is missing data_offsets"))
    })?;
    if array.len() != 2 {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{name}` data_offsets must contain two integers"
        )));
    }
    let start = array[0].as_u64().ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` data_offsets must be integers"))
    })?;
    let end = array[1].as_u64().ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` data_offsets must be integers"))
    })?;
    Ok([start, end])
}

fn usize_from_u64(value: u64, message: &str) -> Result<usize, TensorLoadError> {
    value
        .try_into()
        .map_err(|_| TensorLoadError::integrity(message))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct TensorLoadError {
    code: &'static str,
    message: String,
}

impl TensorLoadError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn integrity(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    fn missing(message: impl Into<String>) -> Self {
        Self {
            code: "model_artifact_missing",
            message: message.into(),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }
}
