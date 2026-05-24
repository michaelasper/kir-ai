use super::{
    NativeTextDiskCacheBlockDescriptor, NativeTextDiskCacheError, NativeTextDiskCacheIdentity,
    NativeTextDiskCacheLayerLayout, NativeTextDiskCacheValue,
};
use serde::{Deserialize, Serialize};

const NATIVE_TEXT_DISK_CACHE_CODEC: &str = "kir-ai-native-text-prefix-block";
const NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION: u32 = 2;
const MAX_SAFETENSORS_HEADER_LEN: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeTextDiskCacheBlockMetadata {
    codec: String,
    layout_version: u32,
    cache_layout_version: u32,
    model_family: String,
    model_hash: String,
    snapshot_hash: String,
    namespace_hash: String,
    previous_block_hash: String,
    block_hash: String,
    block_start: usize,
    token_count: usize,
    hidden_shape: Vec<usize>,
    layers: Vec<NativeTextDiskCacheLayerLayout>,
}

#[derive(Debug)]
pub(crate) struct NativeTextDiskCacheBlock<C: NativeTextDiskCacheValue> {
    pub(crate) block_start: usize,
    pub(crate) token_count: usize,
    pub(crate) hidden: Vec<f32>,
    pub(crate) states: Vec<C::PrefixCacheState>,
}

impl<C> NativeTextDiskCacheBlock<C>
where
    C: NativeTextDiskCacheValue,
{
    pub(crate) fn encode(
        descriptor: &NativeTextDiskCacheBlockDescriptor,
        hidden: &[f32],
        states: &[C::PrefixCacheState],
    ) -> Result<Vec<u8>, NativeTextDiskCacheError> {
        let mut sink = NativeTextDiskCacheTensorSink::default();
        sink.push_f32("hidden", vec![hidden.len()], hidden.to_vec())?;
        let layers = C::encode_disk_block_states(
            states,
            descriptor.block_start,
            descriptor.token_count,
            &mut sink,
        )?;
        let metadata = NativeTextDiskCacheBlockMetadata {
            codec: NATIVE_TEXT_DISK_CACHE_CODEC.to_owned(),
            layout_version: NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION,
            cache_layout_version: descriptor.cache_layout_version,
            model_family: descriptor.model_family.clone(),
            model_hash: descriptor.model_hash.clone(),
            snapshot_hash: descriptor.snapshot_hash.clone(),
            namespace_hash: descriptor.namespace_hash.clone(),
            previous_block_hash: descriptor.previous_block_hash.clone(),
            block_hash: descriptor.block_hash.clone(),
            block_start: descriptor.block_start,
            token_count: descriptor.token_count,
            hidden_shape: vec![hidden.len()],
            layers,
        };
        encode_safetensors(metadata, sink.into_tensors())
    }

    pub(crate) fn decode(
        bytes: &[u8],
        identity: &NativeTextDiskCacheIdentity,
        descriptor: &NativeTextDiskCacheBlockDescriptor,
    ) -> Result<Self, NativeTextDiskCacheError> {
        let (metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        if metadata.namespace_hash != descriptor.namespace_hash
            || metadata.previous_block_hash != descriptor.previous_block_hash
            || metadata.block_hash != descriptor.block_hash
            || metadata.block_start != descriptor.block_start
            || metadata.token_count != descriptor.token_count
        {
            return Err(NativeTextDiskCacheError::integrity(
                "disk cache block descriptor mismatch",
            ));
        }
        descriptor.validate_for_identity(identity)?;
        validate_metadata_for_identity(&metadata, identity)?;
        if metadata.cache_layout_version != descriptor.cache_layout_version {
            return Err(NativeTextDiskCacheError::integrity(
                "cache layout version mismatch",
            ));
        }
        let hidden = archive.f32_tensor("hidden")?;
        if metadata.hidden_shape.as_slice() != [hidden.len()] {
            return Err(NativeTextDiskCacheError::integrity(
                "hidden tensor shape does not match metadata",
            ));
        }
        let states = C::decode_disk_states(&metadata.layers, &archive)?;
        Ok(Self {
            block_start: metadata.block_start,
            token_count: metadata.token_count,
            hidden,
            states,
        })
    }

    pub(super) fn decode_descriptor(
        bytes: &[u8],
    ) -> Result<NativeTextDiskCacheBlockDescriptor, NativeTextDiskCacheError> {
        let metadata = decode_safetensors_metadata(bytes)?;
        validate_metadata_codec_and_layout(&metadata)?;
        Ok(NativeTextDiskCacheBlockDescriptor {
            model_hash: metadata.model_hash,
            snapshot_hash: metadata.snapshot_hash,
            model_family: metadata.model_family,
            namespace_hash: metadata.namespace_hash,
            previous_block_hash: metadata.previous_block_hash,
            block_hash: metadata.block_hash,
            block_start: metadata.block_start,
            token_count: metadata.token_count,
            cache_layout_version: metadata.cache_layout_version,
        })
    }

    #[cfg(test)]
    pub(crate) fn rewrite_layout_version_for_test(
        bytes: &mut Vec<u8>,
        layout_version: u32,
    ) -> Result<(), NativeTextDiskCacheError> {
        let (mut metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        metadata.layout_version = layout_version;
        *bytes = encode_safetensors(metadata, archive.tensors.clone())?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn rewrite_cache_layout_version_for_test(
        bytes: &mut Vec<u8>,
        cache_layout_version: u32,
    ) -> Result<(), NativeTextDiskCacheError> {
        let (mut metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        metadata.cache_layout_version = cache_layout_version;
        *bytes = encode_safetensors(metadata, archive.tensors.clone())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct NativeTextDiskCacheTensorSink {
    tensors: Vec<NativeTextDiskCacheTensor>,
}

impl NativeTextDiskCacheTensorSink {
    pub(crate) fn push_f32(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<(), NativeTextDiskCacheError> {
        let expected = shape.iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor shape overflow"))
        })?;
        if expected != values.len() {
            return Err(NativeTextDiskCacheError::integrity(
                "tensor shape does not match value count",
            ));
        }
        self.tensors.push(NativeTextDiskCacheTensor {
            name: name.into(),
            shape,
            values,
        });
        Ok(())
    }

    fn into_tensors(self) -> Vec<NativeTextDiskCacheTensor> {
        self.tensors
    }
}

#[derive(Debug, Clone)]
struct NativeTextDiskCacheTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct NativeTextDiskCacheTensorArchive<'a> {
    tensors: Vec<NativeTextDiskCacheTensor>,
    bytes: &'a [u8],
}

impl<'a> NativeTextDiskCacheTensorArchive<'a> {
    fn from_bytes(
        bytes: &'a [u8],
    ) -> Result<(NativeTextDiskCacheBlockMetadata, Self), NativeTextDiskCacheError> {
        let header = safetensors_header_json(bytes)?;
        let metadata = metadata_from_header(&header)?;
        let tensors = tensors_from_header(bytes, &header)?;
        let archive = Self { tensors, bytes };
        Ok((metadata, archive))
    }

    pub(crate) fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, NativeTextDiskCacheError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` not found"))
            })?;
        Ok(tensor.values.clone())
    }

    pub(crate) fn f32_tensor_shape(
        &self,
        name: &str,
    ) -> Result<&[usize], NativeTextDiskCacheError> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.shape.as_slice())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` not found"))
            })
    }

    #[allow(dead_code)]
    fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

fn validate_metadata_for_identity(
    metadata: &NativeTextDiskCacheBlockMetadata,
    identity: &NativeTextDiskCacheIdentity,
) -> Result<(), NativeTextDiskCacheError> {
    validate_metadata_codec_and_layout(metadata)?;
    if metadata.model_hash != identity.model_hash {
        return Err(NativeTextDiskCacheError::integrity("model hash mismatch"));
    }
    if metadata.snapshot_hash != identity.snapshot_hash {
        return Err(NativeTextDiskCacheError::integrity(
            "snapshot hash mismatch",
        ));
    }
    if metadata.model_family != identity.model_family {
        return Err(NativeTextDiskCacheError::integrity("model family mismatch"));
    }
    if let Some(namespace_hash) = identity.allowed_namespace_hash.as_deref()
        && metadata.namespace_hash != namespace_hash
    {
        return Err(NativeTextDiskCacheError::integrity(
            "namespace hash mismatch",
        ));
    }
    Ok(())
}

fn validate_metadata_codec_and_layout(
    metadata: &NativeTextDiskCacheBlockMetadata,
) -> Result<(), NativeTextDiskCacheError> {
    if metadata.codec != NATIVE_TEXT_DISK_CACHE_CODEC {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported disk cache codec",
        ));
    }
    if metadata.layout_version != NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported disk cache layout version",
        ));
    }
    Ok(())
}

fn encode_safetensors(
    metadata: NativeTextDiskCacheBlockMetadata,
    tensors: Vec<NativeTextDiskCacheTensor>,
) -> Result<Vec<u8>, NativeTextDiskCacheError> {
    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "kir_ai": serde_json::to_string(&metadata).map_err(|err| {
                NativeTextDiskCacheError::integrity(format!("disk cache metadata encode failed: {err}"))
            })?,
        }),
    );
    let mut sorted = tensors;
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    for tensor in &sorted {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let end = payload.len();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("safetensors header encode failed: {err}"))
    })?;
    let header_len: u64 = header_bytes
        .len()
        .try_into()
        .map_err(|_| NativeTextDiskCacheError::integrity("safetensors header too large"))?;
    let mut bytes = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn decode_safetensors_metadata(
    bytes: &[u8],
) -> Result<NativeTextDiskCacheBlockMetadata, NativeTextDiskCacheError> {
    metadata_from_header(&safetensors_header_json(bytes)?)
}

fn safetensors_header_json(
    bytes: &[u8],
) -> Result<serde_json::Map<String, serde_json::Value>, NativeTextDiskCacheError> {
    let header_len_bytes = bytes.get(0..8).ok_or_else(|| {
        NativeTextDiskCacheError::integrity("safetensors file is missing header length")
    })?;
    let header_len = u64::from_le_bytes(
        header_len_bytes
            .try_into()
            .map_err(|_| NativeTextDiskCacheError::integrity("header length is not 8 bytes"))?,
    );
    let header_len: usize = header_len
        .try_into()
        .map_err(|_| NativeTextDiskCacheError::integrity("safetensors header too large"))?;
    if header_len > MAX_SAFETENSORS_HEADER_LEN {
        return Err(NativeTextDiskCacheError::integrity(
            "safetensors header exceeds limit",
        ));
    }
    let header_end = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header overflow"))?;
    let header = bytes
        .get(8..header_end)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header is truncated"))?;
    let value: serde_json::Value = serde_json::from_slice(header).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("invalid safetensors header json: {err}"))
    })?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header is not an object"))
}

fn metadata_from_header(
    header: &serde_json::Map<String, serde_json::Value>,
) -> Result<NativeTextDiskCacheBlockMetadata, NativeTextDiskCacheError> {
    let metadata = header
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .and_then(|metadata| metadata.get("kir_ai"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("missing disk cache metadata"))?;
    serde_json::from_str(metadata).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("invalid disk cache metadata: {err}"))
    })
}

fn tensors_from_header(
    bytes: &[u8],
    header: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<NativeTextDiskCacheTensor>, NativeTextDiskCacheError> {
    let header_len = u64::from_le_bytes(
        bytes
            .get(0..8)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("missing header length"))?
            .try_into()
            .map_err(|_| NativeTextDiskCacheError::integrity("header length is not 8 bytes"))?,
    ) as usize;
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors data start overflow"))?;
    let mut tensors = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let object = value.as_object().ok_or_else(|| {
            NativeTextDiskCacheError::integrity(format!("tensor `{name}` header is not an object"))
        })?;
        if object.get("dtype").and_then(serde_json::Value::as_str) != Some("F32") {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` is not F32"
            )));
        }
        let shape = object
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` is missing shape"))
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        NativeTextDiskCacheError::integrity(format!(
                            "tensor `{name}` has invalid shape"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = object
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!(
                    "tensor `{name}` is missing data offsets"
                ))
            })?;
        if offsets.len() != 2 {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` data offsets are invalid"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` offset is invalid"))
            })?;
        let end = offsets[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` offset is invalid"))
            })?;
        if end < start || (end - start) % std::mem::size_of::<f32>() != 0 {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` byte range is invalid"
            )));
        }
        let file_start = data_start
            .checked_add(start)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor start overflow"))?;
        let file_end = data_start
            .checked_add(end)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor end overflow"))?;
        let data = bytes.get(file_start..file_end).ok_or_else(|| {
            NativeTextDiskCacheError::integrity(format!("tensor `{name}` payload is truncated"))
        })?;
        let values = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        tensors.push(NativeTextDiskCacheTensor {
            name: name.clone(),
            shape,
            values,
        });
    }
    if tensors.is_empty() {
        return Err(NativeTextDiskCacheError::integrity(
            "disk cache block has no tensors",
        ));
    }
    Ok(tensors)
}
