use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub type TinyBf16Tensor = (String, Vec<usize>, Vec<f32>);

pub trait TinyBf16TensorRef {
    fn name(&self) -> &str;
    fn shape(&self) -> &[usize];
    fn values(&self) -> &[f32];
}

impl TinyBf16TensorRef for (String, Vec<usize>, Vec<f32>) {
    fn name(&self) -> &str {
        &self.0
    }

    fn shape(&self) -> &[usize] {
        &self.1
    }

    fn values(&self) -> &[f32] {
        &self.2
    }
}

impl TinyBf16TensorRef for (&str, Vec<usize>, Vec<f32>) {
    fn name(&self) -> &str {
        self.0
    }

    fn shape(&self) -> &[usize] {
        &self.1
    }

    fn values(&self) -> &[f32] {
        &self.2
    }
}

#[derive(Clone, Debug, Default)]
pub struct TinySafetensorsSnapshot {
    tensors: Vec<TinySafetensorsShardTensor>,
}

#[derive(Clone, Debug)]
struct TinySafetensorsShardTensor {
    filename: String,
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WrittenTinySafetensorsSnapshot {
    pub total_size: usize,
    pub tensor_count: usize,
    pub shard_count: usize,
}

impl TinySafetensorsSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_bf16_tensor(
        mut self,
        filename: impl Into<String>,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: impl Into<Vec<f32>>,
    ) -> Self {
        self.push_bf16_tensor(filename, name, shape, values);
        self
    }

    pub fn push_bf16_tensor(
        &mut self,
        filename: impl Into<String>,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: impl Into<Vec<f32>>,
    ) -> &mut Self {
        self.tensors.push(TinySafetensorsShardTensor {
            filename: filename.into(),
            name: name.into(),
            shape: shape.into(),
            values: values.into(),
        });
        self
    }

    pub fn write(&self, root: impl AsRef<Path>) -> io::Result<WrittenTinySafetensorsSnapshot> {
        let root = root.as_ref();
        std::fs::create_dir_all(root)?;

        let mut shards: BTreeMap<String, Vec<TinyBf16Tensor>> = BTreeMap::new();
        for tensor in &self.tensors {
            shards.entry(tensor.filename.clone()).or_default().push((
                tensor.name.clone(),
                tensor.shape.clone(),
                tensor.values.clone(),
            ));
        }

        let mut total_size = 0;
        for (filename, tensors) in &shards {
            let path = root.join(filename);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let bytes = tiny_owned_multi_safetensors_bf16(tensors);
            total_size += bytes.len();
            std::fs::write(path, bytes)?;
        }

        let index_entries = self
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.filename.as_str()));
        std::fs::write(
            root.join("model.safetensors.index.json"),
            tiny_safetensors_index_json(total_size, index_entries),
        )?;

        Ok(WrittenTinySafetensorsSnapshot {
            total_size,
            tensor_count: self.tensors.len(),
            shard_count: shards.len(),
        })
    }
}

pub fn temp_snapshot_dir(prefix: &str, label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{label}-{}", std::process::id()))
}

pub fn tiny_safetensors_index_json<'a>(
    total_size: usize,
    weight_map: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    let weight_map = weight_map
        .into_iter()
        .map(|(tensor, shard)| {
            (
                tensor.to_owned(),
                serde_json::Value::String(shard.to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map
    })
    .to_string()
}

pub fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
}

pub fn tiny_safetensors_f32(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    tiny_safetensors(name, "F32", shape, &data)
}

pub fn tiny_multi_safetensors_bf16(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in *values {
            data.extend_from_slice(&bf16_bits(*value).to_le_bytes());
        }
        let end = data.len();
        insert_tensor_header(&mut header, name, "BF16", shape, start, end);
    }
    safetensors_with_header(header, &data)
}

pub fn tiny_owned_multi_safetensors_bf16<T: TinyBf16TensorRef>(tensors: &[T]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        for value in tensor.values() {
            data.extend_from_slice(&bf16_bits(*value).to_le_bytes());
        }
        let end = data.len();
        insert_tensor_header(
            &mut header,
            tensor.name(),
            "BF16",
            tensor.shape(),
            start,
            end,
        );
    }
    safetensors_with_header(header, &data)
}

pub fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    insert_tensor_header(&mut header, name, dtype, shape, 0, data.len());
    safetensors_with_header(header, data)
}

pub fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn insert_tensor_header(
    header: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
    dtype: &str,
    shape: &[usize],
    start: usize,
    end: usize,
) {
    header.insert(
        name.to_owned(),
        serde_json::json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end]
        }),
    );
}

fn safetensors_with_header(
    header: serde_json::Map<String, serde_json::Value>,
    data: &[u8],
) -> Vec<u8> {
    let header = serde_json::Value::Object(header).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}
