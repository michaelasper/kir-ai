mod header;
mod math;

use self::{
    header::{parse_shape, read_header_prefix, validate_header_len},
    math::{
        bf16_bytes_to_bits, bf16_bytes_to_f32_into, bf16_dot_f32, bf16_row_byte_len, bf16_row_bytes,
    },
};
use super::math::{TopKLogit, push_top_logit};
use super::native_matvec::NativeBatchedMatvecOutput;
use llm_models::SafetensorsIndex;
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use thiserror::Error;

const BF16_MATVEC_CHUNK_ROWS: usize = 256;

#[cfg(test)]
mod mmap_materialization_test_hook {
    use std::{
        path::Path,
        sync::{Arc, Mutex, OnceLock},
    };

    type Hook = Arc<dyn Fn(Option<&Path>) + Send + Sync + 'static>;

    static HOOK: OnceLock<Mutex<Option<Hook>>> = OnceLock::new();

    pub(super) struct HookGuard;

    pub(super) fn set(hook: Hook) -> HookGuard {
        let mut slot = hook_slot()
            .lock()
            .expect("mmap materialization test hook lock");
        assert!(
            slot.replace(hook).is_none(),
            "mmap materialization test hook already installed"
        );
        HookGuard
    }

    pub(super) fn notify(source_path: Option<&Path>) {
        let hook = hook_slot()
            .lock()
            .expect("mmap materialization test hook lock")
            .as_ref()
            .map(Arc::clone);
        if let Some(hook) = hook {
            hook(source_path);
        }
    }

    fn hook_slot() -> &'static Mutex<Option<Hook>> {
        HOOK.get_or_init(|| Mutex::new(None))
    }

    impl Drop for HookGuard {
        fn drop(&mut self) {
            *hook_slot()
                .lock()
                .expect("mmap materialization test hook lock") = None;
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorArchive {
    bytes: Box<[u8]>,
}

impl SafeTensorArchive {
    /// Builds an archive from borrowed safetensors bytes.
    ///
    /// The input is validated before being copied into archive-owned storage.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorLoadError> {
        Self::validate_bytes(bytes)?;
        Ok(Self {
            bytes: bytes.into(),
        })
    }

    /// Builds an archive from owned boxed safetensors bytes without copying.
    ///
    /// The owned input is validated before the same boxed byte slice is stored.
    pub fn from_owned_bytes(bytes: Box<[u8]>) -> Result<Self, TensorLoadError> {
        Self::validate_bytes(bytes.as_ref())?;
        Ok(Self { bytes })
    }

    fn validate_bytes(bytes: &[u8]) -> Result<(), TensorLoadError> {
        SafeTensors::deserialize(bytes)
            .map_err(|err| TensorLoadError::integrity(format!("invalid safetensors: {err}")))?;
        Ok(())
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
        data.chunks_exact(4)
            .map(|chunk| {
                chunk
                    .try_into()
                    .map(f32::from_le_bytes)
                    .map_err(|_| TensorLoadError::integrity("f32 tensor chunk is not 4 bytes"))
            })
            .collect::<Result<_, _>>()
    }

    fn tensors(&self) -> Result<SafeTensors<'_>, TensorLoadError> {
        SafeTensors::deserialize(self.bytes.as_ref())
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

#[derive(Debug)]
pub struct SafeTensorFile {
    header: SafeTensorHeader,
    file: File,
    mapped: Mutex<Option<Arc<Mmap>>>,
}

impl SafeTensorFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let path = path.as_ref();
        let header = SafeTensorHeader::from_file(path)?;
        let file = File::open(path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not open safetensors file `{}`: {err}",
                path.display()
            ))
        })?;
        Ok(Self {
            header,
            file,
            mapped: Mutex::new(None),
        })
    }

    pub fn header(&self) -> &SafeTensorHeader {
        &self.header
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        self.header.tensor_metadata(name)
    }

    pub fn materialize(&self) -> Result<usize, TensorLoadError> {
        let mapped = self.materialized_file()?;
        Ok(mapped.len())
    }

    pub fn is_materialized(&self) -> bool {
        self.mapped
            .lock()
            .map(|mapped| mapped.is_some())
            .unwrap_or(false)
    }

    pub fn tensor_bytes_range(
        &self,
        name: &str,
        tensor_byte_offset: u64,
        byte_len: usize,
    ) -> Result<Vec<u8>, TensorLoadError> {
        let file_range = self.tensor_file_byte_range(name, tensor_byte_offset, byte_len)?;
        if let Some(mapped) = self.materialized_file_if_present()? {
            let bytes = mapped.get(file_range.clone()).ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` mapped range is invalid"))
            })?;
            return Ok(bytes.to_vec());
        }
        self.read_tensor_bytes_range(name, file_range, byte_len)
    }

    pub fn with_tensor_bytes_range<T>(
        &self,
        name: &str,
        tensor_byte_offset: u64,
        byte_len: usize,
        read: impl FnOnce(&[u8]) -> Result<T, TensorLoadError>,
    ) -> Result<T, TensorLoadError> {
        let file_range = self.tensor_file_byte_range(name, tensor_byte_offset, byte_len)?;
        if let Some(mapped) = self.materialized_file_if_present()? {
            let bytes = mapped.get(file_range).ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` mapped range is invalid"))
            })?;
            return read(bytes);
        }
        let bytes = self.read_tensor_bytes_range(name, file_range, byte_len)?;
        read(&bytes)
    }

    fn read_tensor_bytes_range(
        &self,
        name: &str,
        file_range: Range<usize>,
        byte_len: usize,
    ) -> Result<Vec<u8>, TensorLoadError> {
        let mut bytes = vec![0_u8; byte_len];
        let mut file = self.file.try_clone().map_err(|err| {
            TensorLoadError::integrity(format!("could not clone safetensors file handle: {err}"))
        })?;
        file.seek(SeekFrom::Start(file_range.start as u64))
            .map_err(|err| {
                TensorLoadError::integrity(format!("could not seek tensor `{name}`: {err}"))
            })?;
        file.read_exact(&mut bytes).map_err(|err| {
            TensorLoadError::integrity(format!("could not read tensor `{name}` bytes: {err}"))
        })?;
        Ok(bytes)
    }

    fn tensor_file_byte_range(
        &self,
        name: &str,
        tensor_byte_offset: u64,
        byte_len: usize,
    ) -> Result<Range<usize>, TensorLoadError> {
        let tensor = self.header.tensor_entry(name)?;
        let tensor_byte_len = u64_from_usize(
            tensor.byte_len()?,
            "tensor byte length does not fit in u64 for range read",
        )?;
        let requested_end = tensor_byte_offset
            .checked_add(u64_from_usize(
                byte_len,
                "requested byte length does not fit in u64",
            )?)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` range overflow")))?;
        if requested_end > tensor_byte_len {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` requested byte range exceeds tensor length"
            )));
        }
        let tensor_range = self.header.tensor_data_range(name)?;
        let file_offset = tensor_range
            .start
            .checked_add(tensor_byte_offset)
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        let file_end = file_offset
            .checked_add(u64_from_usize(
                byte_len,
                "requested byte length does not fit in u64",
            )?)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` range overflow")))?;
        Ok(
            usize_from_u64(file_offset, "tensor file offset does not fit in usize")?
                ..usize_from_u64(file_end, "tensor file end does not fit in usize")?,
        )
    }

    fn materialized_file(&self) -> Result<Arc<Mmap>, TensorLoadError> {
        let mut cached = self.mapped.lock().map_err(|err| {
            TensorLoadError::integrity(format!("mmap cache lock poisoned: {err}"))
        })?;
        if let Some(mapped) = cached.as_ref() {
            return Ok(Arc::clone(mapped));
        }
        let expected_len = usize_from_u64(
            self.header.file_len(),
            "safetensors file length does not fit in usize for mmap",
        )?;
        #[cfg(test)]
        mmap_materialization_test_hook::notify(self.header.source_path());
        // SAFETY: promoted safetensors snapshots are treated as immutable by the
        // store. This read-only mapping is used only after header/range validation,
        // and callers borrow or copy validated byte ranges before decoding.
        let mapped = Arc::new(
            unsafe { MmapOptions::new().map(&self.file) }.map_err(|err| {
                TensorLoadError::integrity(format!("could not mmap safetensors file: {err}"))
            })?,
        );
        if mapped.len() != expected_len {
            return Err(TensorLoadError::integrity(format!(
                "mmap length {} does not match safetensors header length {expected_len}",
                mapped.len()
            )));
        }
        *cached = Some(Arc::clone(&mapped));
        Ok(mapped)
    }

    fn materialized_file_if_present(&self) -> Result<Option<Arc<Mmap>>, TensorLoadError> {
        self.mapped
            .lock()
            .map(|mapped| mapped.as_ref().map(Arc::clone))
            .map_err(|err| TensorLoadError::integrity(format!("mmap cache lock poisoned: {err}")))
    }

    pub fn bf16_tensor_f32_range(
        &self,
        name: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let mut values = Vec::with_capacity(element_count);
        self.bf16_tensor_f32_range_into(name, element_offset, element_count, &mut values)?;
        Ok(values)
    }

    pub fn bf16_tensor_f32_range_into(
        &self,
        name: &str,
        element_offset: usize,
        element_count: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TensorLoadError> {
        self.with_bf16_tensor_bytes_range(name, element_offset, element_count, |bytes| {
            bf16_bytes_to_f32_into(bytes, output)
        })
    }

    fn with_bf16_tensor_bytes_range<T>(
        &self,
        name: &str,
        element_offset: usize,
        element_count: usize,
        read: impl FnOnce(&[u8]) -> Result<T, TensorLoadError>,
    ) -> Result<T, TensorLoadError> {
        let tensor = self.header.tensor_entry(name)?;
        if tensor.dtype != "BF16" {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` has dtype {}, expected BF16",
                tensor.dtype
            )));
        }
        let byte_offset = u64_from_usize(
            element_offset
                .checked_mul(2)
                .ok_or_else(|| TensorLoadError::integrity("BF16 element offset overflow"))?,
            "BF16 byte offset does not fit in u64",
        )?;
        let byte_len = element_count
            .checked_mul(2)
            .ok_or_else(|| TensorLoadError::integrity("BF16 element count overflow"))?;
        self.with_tensor_bytes_range(name, byte_offset, byte_len, read)
    }

    pub fn bf16_tensor_bits_range(
        &self,
        name: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<u16>, TensorLoadError> {
        self.with_bf16_tensor_bytes_range(name, element_offset, element_count, bf16_bytes_to_bits)
    }

    pub fn bf16_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, TensorLoadError> {
        let tensor = self.header.tensor_entry(name)?;
        if tensor.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` row reader expects rank 2, got rank {}",
                tensor.shape.len()
            )));
        }
        let rows = tensor.shape[0];
        let columns = tensor.shape[1];
        if row >= rows {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` row {row} exceeds row count {rows}"
            )));
        }
        let mut values = Vec::with_capacity(columns);
        self.bf16_row_f32_into(name, row, &mut values)?;
        Ok(values)
    }

    pub fn bf16_row_f32_into(
        &self,
        name: &str,
        row: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TensorLoadError> {
        let tensor = self.header.tensor_entry(name)?;
        if tensor.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` row reader expects rank 2, got rank {}",
                tensor.shape.len()
            )));
        }
        let rows = tensor.shape[0];
        let columns = tensor.shape[1];
        if row >= rows {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` row {row} exceeds row count {rows}"
            )));
        }
        let element_offset = row
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("row offset overflow"))?;
        self.bf16_tensor_f32_range_into(name, element_offset, columns, output)
    }
}

/// Summary for the compatibility warmup path.
///
/// COR-454 removed the permanent BF16→F32 weight cache. Warmup still validates
/// the requested static tensor set, but it does not load or retain F32 storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct F32TensorCacheWarmup {
    pub candidates: u64,
    pub loaded: u64,
    pub already_resident: u64,
    pub resident_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct SafeTensorShardStore {
    root: PathBuf,
    index: SafetensorsIndex,
    shards: Arc<Mutex<BTreeMap<PathBuf, Arc<SafeTensorFile>>>>,
}

impl SafeTensorShardStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join("model.safetensors.index.json");
        let index = match fs::read_to_string(&index_path) {
            Ok(index_json) => SafetensorsIndex::from_json(&index_json).map_err(|err| {
                TensorLoadError::integrity(format!(
                    "invalid safetensors index `{}`: {err}",
                    index_path.display()
                ))
            })?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let shard_name = "model.safetensors";
                let shard_path = root.join(shard_name);
                let header = SafeTensorHeader::from_file(&shard_path)?;
                SafetensorsIndex::single_file(
                    header.file_len(),
                    shard_name,
                    header.tensor_names().map(str::to_owned),
                )
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "invalid single-file safetensors snapshot `{}`: {err}",
                        shard_path.display()
                    ))
                })?
            }
            Err(err) => {
                return Err(TensorLoadError::missing(format!(
                    "could not read safetensors index `{}`: {err}",
                    index_path.display()
                )));
            }
        };
        Ok(Self {
            root,
            index,
            shards: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn tensor_shard_path(&self, tensor: &str) -> Result<PathBuf, TensorLoadError> {
        let shard = self.index.shard_for(tensor).ok_or_else(|| {
            TensorLoadError::missing(format!("tensor `{tensor}` not found in safetensors index"))
        })?;
        self.resolve_shard_path(shard)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.tensor_names()
    }

    pub fn index(&self) -> &SafetensorsIndex {
        &self.index
    }

    fn resolve_shard_path(&self, shard: &str) -> Result<PathBuf, TensorLoadError> {
        let root = fs::canonicalize(&self.root).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not resolve safetensors snapshot root `{}`: {err}",
                self.root.display()
            ))
        })?;
        let path = root.join(shard);
        let path = fs::canonicalize(&path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not resolve safetensors shard `{}`: {err}",
                path.display()
            ))
        })?;
        if !path.starts_with(&root) {
            return Err(TensorLoadError::integrity(format!(
                "safetensors shard `{}` escapes snapshot root `{}`",
                path.display(),
                root.display()
            )));
        }
        Ok(path)
    }

    pub fn tensor_metadata(&self, tensor: &str) -> Result<TensorMetadata, TensorLoadError> {
        self.open_tensor_file(tensor)?.tensor_metadata(tensor)
    }

    pub fn bf16_row_f32(&self, tensor: &str, row: usize) -> Result<Vec<f32>, TensorLoadError> {
        self.open_tensor_file(tensor)?.bf16_row_f32(tensor, row)
    }

    pub fn bf16_row_f32_into(
        &self,
        tensor: &str,
        row: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TensorLoadError> {
        self.open_tensor_file(tensor)?
            .bf16_row_f32_into(tensor, row, output)
    }

    pub fn bf16_tensor_f32_range(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.open_tensor_file(tensor)?
            .bf16_tensor_f32_range(tensor, element_offset, element_count)
    }

    pub fn bf16_tensor_f32_range_into(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TensorLoadError> {
        self.open_tensor_file(tensor)?.bf16_tensor_f32_range_into(
            tensor,
            element_offset,
            element_count,
            output,
        )
    }

    pub fn bf16_tensor_bits_range(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<u16>, TensorLoadError> {
        self.open_tensor_file(tensor)?
            .bf16_tensor_bits_range(tensor, element_offset, element_count)
    }

    pub fn bf16_tensor_f32(&self, tensor: &str) -> Result<Vec<f32>, TensorLoadError> {
        let metadata = self.tensor_metadata(tensor)?;
        let element_count = metadata.shape.iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| TensorLoadError::integrity("tensor shape overflows usize"))
        })?;
        self.bf16_tensor_f32_range(tensor, 0, element_count)
    }

    pub fn bf16_tensor_f32_range_cached(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        tracing::trace!(
            operation = "safetensors_f32_cache_bypass",
            cache = "range",
            cache_resident = false,
            tensor = tensor,
            element_offset,
            element_count,
            "safetensors F32 range cache bypass; decoding on demand"
        );
        self.bf16_tensor_f32_range(tensor, element_offset, element_count)
    }

    pub fn bf16_tensor_f32_range_cached_arc(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Arc<[f32]>, TensorLoadError> {
        tracing::trace!(
            operation = "safetensors_f32_cache_bypass",
            cache = "range",
            cache_resident = false,
            tensor = tensor,
            element_offset,
            element_count,
            "safetensors F32 range cache bypass; decoding transient Arc on demand"
        );
        let values = self.bf16_tensor_f32_range(tensor, element_offset, element_count)?;
        Ok(values.into_boxed_slice().into())
    }

    pub fn bf16_tensor_f32_cached(&self, tensor: &str) -> Result<Vec<f32>, TensorLoadError> {
        tracing::trace!(
            operation = "safetensors_f32_cache_bypass",
            cache = "full",
            cache_resident = false,
            tensor = tensor,
            "safetensors F32 full tensor cache bypass; decoding on demand"
        );
        self.bf16_tensor_f32(tensor)
    }

    pub fn bf16_tensor_f32_cached_arc(&self, tensor: &str) -> Result<Arc<[f32]>, TensorLoadError> {
        tracing::trace!(
            operation = "safetensors_f32_cache_bypass",
            cache = "full",
            cache_resident = false,
            tensor = tensor,
            "safetensors F32 full tensor cache bypass; decoding transient Arc on demand"
        );
        let values = self.bf16_tensor_f32(tensor)?;
        Ok(values.into_boxed_slice().into())
    }

    pub fn preload_bf16_f32_tensors(
        &self,
        tensors: &[String],
    ) -> Result<F32TensorCacheWarmup, TensorLoadError> {
        let mut seen = BTreeSet::new();
        let mut warmup = F32TensorCacheWarmup::default();
        for tensor in tensors {
            if !seen.insert(tensor.as_str()) {
                continue;
            }
            warmup.candidates += 1;
            let metadata = self.tensor_metadata(tensor)?;
            if metadata.dtype != "BF16" {
                return Err(TensorLoadError::unsupported(format!(
                    "tensor `{tensor}` has dtype {}, expected BF16",
                    metadata.dtype
                )));
            }
            metadata.shape.iter().try_fold(1_usize, |acc, dim| {
                acc.checked_mul(*dim)
                    .ok_or_else(|| TensorLoadError::integrity("tensor shape overflows usize"))
            })?;
        }
        tracing::trace!(
            operation = "safetensors_f32_cache_warmup_bypass",
            candidates = warmup.candidates,
            loaded = warmup.loaded,
            already_resident = warmup.already_resident,
            resident_bytes = warmup.resident_bytes,
            "safetensors F32 warmup skipped permanent resident cache"
        );
        Ok(warmup)
    }

    pub fn cached_f32_count(&self) -> usize {
        0
    }

    pub fn cached_f32_bytes(&self) -> u64 {
        0
    }

    pub fn bf16_matvec_row_major_f32(
        &self,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.bf16_matvec_rows_f32(tensor, input, BF16_MATVEC_CHUNK_ROWS)
    }

    pub fn bf16_matvec_row_major_f32_in_place(
        &self,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        self.bf16_matvec_rows_f32_in_place(tensor, input, BF16_MATVEC_CHUNK_ROWS, output)
    }

    pub fn bf16_matvec_rows_f32(
        &self,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let entry = file.header.tensor_entry(tensor)?;
        if entry.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` matvec expects rank 2, got rank {}",
                entry.shape.len()
            )));
        }
        let rows = entry.shape[0];
        let columns = entry.shape[1];
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "input length {} does not match tensor `{tensor}` columns {columns}",
                input.len()
            )));
        }
        if chunk_rows == 0 {
            return Err(TensorLoadError::integrity(
                "chunk_rows must be greater than zero",
            ));
        }
        let row_byte_len = bf16_row_byte_len(columns, "matvec")?;
        let mut output = Vec::with_capacity(rows);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec chunk overflow"))?;
            file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
                for row_offset in 0..rows_in_chunk {
                    let row = bf16_row_bytes(weights, row_offset, row_byte_len, "matvec chunk")?;
                    output.push(bf16_dot_f32(row, input)?);
                }
                Ok(())
            })?;
        }
        Ok(output)
    }

    pub fn bf16_matvec_rows_f32_in_place(
        &self,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let entry = file.header.tensor_entry(tensor)?;
        if entry.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` matvec expects rank 2, got rank {}",
                entry.shape.len()
            )));
        }
        let rows = entry.shape[0];
        let columns = entry.shape[1];
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "input length {} does not match tensor `{tensor}` columns {columns}",
                input.len()
            )));
        }
        if output.len() < rows {
            return Err(TensorLoadError::integrity(
                "output buffer too small for BF16 matvec",
            ));
        }
        if chunk_rows == 0 {
            return Err(TensorLoadError::integrity(
                "chunk_rows must be greater than zero",
            ));
        }
        let row_byte_len = bf16_row_byte_len(columns, "matvec")?;
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec chunk overflow"))?;
            file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
                for row_offset in 0..rows_in_chunk {
                    let row = bf16_row_bytes(weights, row_offset, row_byte_len, "matvec chunk")?;
                    output[row_start + row_offset] = bf16_dot_f32(row, input)?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bf16_matvec_range_row_major_f32_in_place(
        &self,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "BF16 range matvec input length {} must match columns {columns}",
                input.len()
            )));
        }
        if output.len() < rows {
            return Err(TensorLoadError::integrity(
                "BF16 range matvec failed: output buffer too small",
            ));
        }
        let element_count = rows
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("BF16 range matvec shape overflow"))?;
        let row_byte_len = bf16_row_byte_len(columns, "BF16 range matvec")?;
        let file = self.open_tensor_file(tensor)?;
        file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
            for (row_offset, out) in output.iter_mut().take(rows).enumerate() {
                let row =
                    bf16_row_bytes(weights, row_offset, row_byte_len, "BF16 range matvec chunk")?;
                *out = bf16_dot_f32(row, input)?;
            }
            Ok(())
        })
    }

    pub fn bf16_matvecs_row_major_f32(
        &self,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        Ok(self
            .bf16_matvecs_row_major_f32_flat(tensor, inputs)?
            .into_rows())
    }

    pub fn bf16_matvecs_row_major_f32_flat(
        &self,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<NativeBatchedMatvecOutput, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let entry = file.header.tensor_entry(tensor)?;
        if entry.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` batched matvec expects rank 2, got rank {}",
                entry.shape.len()
            )));
        }
        let rows = entry.shape[0];
        let columns = entry.shape[1];
        for input in inputs {
            if input.len() != columns {
                return Err(TensorLoadError::integrity(format!(
                    "input length {} does not match tensor `{tensor}` columns {columns}",
                    input.len()
                )));
            }
        }
        let output_len = inputs
            .len()
            .checked_mul(rows)
            .ok_or_else(|| TensorLoadError::integrity("batched matvec output overflow"))?;
        let mut outputs = vec![0.0; output_len];
        for row_start in (0..rows).step_by(BF16_MATVEC_CHUNK_ROWS) {
            let rows_in_chunk = BF16_MATVEC_CHUNK_ROWS.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec chunk overflow"))?;
            let row_byte_len = bf16_row_byte_len(columns, "batched matvec")?;
            file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
                for row_offset in 0..rows_in_chunk {
                    let row =
                        bf16_row_bytes(weights, row_offset, row_byte_len, "batched matvec chunk")?;
                    for (input_idx, input) in inputs.iter().enumerate() {
                        let output_index = input_idx
                            .checked_mul(rows)
                            .and_then(|start| start.checked_add(row_start))
                            .and_then(|start| start.checked_add(row_offset))
                            .ok_or_else(|| {
                                TensorLoadError::integrity("batched matvec output offset overflow")
                            })?;
                        outputs[output_index] = bf16_dot_f32(row, input)?;
                    }
                }
                Ok(())
            })?;
        }
        NativeBatchedMatvecOutput::new(outputs, rows)
    }

    pub fn bf16_matvecs_row_major_f32_flat_inputs(
        &self,
        tensor: &str,
        inputs: &[f32],
        input_count: usize,
    ) -> Result<NativeBatchedMatvecOutput, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let entry = file.header.tensor_entry(tensor)?;
        if entry.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` flat-input batched matvec expects rank 2, got rank {}",
                entry.shape.len()
            )));
        }
        let rows = entry.shape[0];
        let columns = entry.shape[1];
        let expected_inputs_len = input_count.checked_mul(columns).ok_or_else(|| {
            TensorLoadError::integrity("flat-input batched matvec input shape overflow")
        })?;
        if inputs.len() != expected_inputs_len {
            return Err(TensorLoadError::integrity(format!(
                "flat input length {} does not match input_count {input_count} * tensor `{tensor}` columns {columns}",
                inputs.len()
            )));
        }
        let output_len = input_count
            .checked_mul(rows)
            .ok_or_else(|| TensorLoadError::integrity("batched matvec output overflow"))?;
        let mut outputs = vec![0.0; output_len];
        if columns == 0 {
            return NativeBatchedMatvecOutput::new(outputs, rows);
        }
        for row_start in (0..rows).step_by(BF16_MATVEC_CHUNK_ROWS) {
            let rows_in_chunk = BF16_MATVEC_CHUNK_ROWS.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec chunk overflow"))?;
            let row_byte_len = bf16_row_byte_len(columns, "flat-input batched matvec")?;
            file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
                for row_offset in 0..rows_in_chunk {
                    let row = bf16_row_bytes(
                        weights,
                        row_offset,
                        row_byte_len,
                        "flat-input batched matvec chunk",
                    )?;
                    for (input_idx, input) in inputs.chunks_exact(columns).enumerate() {
                        let output_index = input_idx
                            .checked_mul(rows)
                            .and_then(|start| start.checked_add(row_start))
                            .and_then(|start| start.checked_add(row_offset))
                            .ok_or_else(|| {
                                TensorLoadError::integrity("batched matvec output offset overflow")
                            })?;
                        outputs[output_index] = bf16_dot_f32(row, input)?;
                    }
                }
                Ok(())
            })?;
        }
        NativeBatchedMatvecOutput::new(outputs, rows)
    }

    pub fn bf16_matvec_top_k_rows_f32(
        &self,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let entry = file.header.tensor_entry(tensor)?;
        if entry.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` top-k matvec expects rank 2, got rank {}",
                entry.shape.len()
            )));
        }
        let rows = entry.shape[0];
        let columns = entry.shape[1];
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "input length {} does not match tensor `{tensor}` columns {columns}",
                input.len()
            )));
        }
        if top_k == 0 || top_k > rows {
            return Err(TensorLoadError::integrity(format!(
                "top_k {top_k} must be in 1..={rows}"
            )));
        }
        if chunk_rows == 0 {
            return Err(TensorLoadError::integrity(
                "chunk_rows must be greater than zero",
            ));
        }
        let row_byte_len = bf16_row_byte_len(columns, "top-k matvec")?;
        let mut top = Vec::with_capacity(top_k);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("top-k matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("top-k matvec chunk overflow"))?;
            file.with_bf16_tensor_bytes_range(tensor, element_offset, element_count, |weights| {
                for row_offset in 0..rows_in_chunk {
                    let row =
                        bf16_row_bytes(weights, row_offset, row_byte_len, "top-k matvec chunk")?;
                    let logit = bf16_dot_f32(row, input)?;
                    push_top_logit(
                        &mut top,
                        TopKLogit {
                            index: row_start + row_offset,
                            logit,
                        },
                        top_k,
                    )
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
                }
                Ok(())
            })?;
        }
        Ok(top)
    }

    pub fn cached_shard_count(&self) -> usize {
        self.shards.lock().map(|shards| shards.len()).unwrap_or(0)
    }

    pub fn materialized_shard_count(&self) -> usize {
        self.shards
            .lock()
            .map(|shards| {
                shards
                    .values()
                    .filter(|shard| shard.is_materialized())
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn materialize_shard_for_tensor(&self, tensor: &str) -> Result<usize, TensorLoadError> {
        self.open_tensor_file(tensor)?.materialize()
    }

    pub fn materialize_all_shards(&self) -> Result<usize, TensorLoadError> {
        let mut total_bytes = 0_usize;
        for shard in self.index.shard_paths() {
            let shard_path = self.resolve_shard_path(shard)?;
            let file = self.open_shard_file(shard_path)?;
            total_bytes = total_bytes
                .checked_add(file.materialize()?)
                .ok_or_else(|| TensorLoadError::integrity("materialized shard bytes overflow"))?;
        }
        Ok(total_bytes)
    }

    fn open_tensor_file(&self, tensor: &str) -> Result<Arc<SafeTensorFile>, TensorLoadError> {
        let shard_path = self.tensor_shard_path(tensor)?;
        self.open_shard_file(shard_path)
    }

    fn open_shard_file(&self, shard_path: PathBuf) -> Result<Arc<SafeTensorFile>, TensorLoadError> {
        {
            let shards = self.shards.lock().map_err(|err| {
                TensorLoadError::integrity(format!("shard cache lock poisoned: {err}"))
            })?;
            if let Some(file) = shards.get(&shard_path) {
                tracing::trace!(
                    operation = "safetensors_shard_cache_lookup",
                    cache_hit = true,
                    shard_path = %shard_path.display(),
                    "safetensors shard cache hit"
                );
                return Ok(Arc::clone(file));
            }
        }
        tracing::trace!(
            operation = "safetensors_shard_cache_lookup",
            cache_hit = false,
            shard_path = %shard_path.display(),
            "safetensors shard cache miss"
        );
        let file = Arc::new(SafeTensorFile::open(&shard_path)?);
        let mut shards = self.shards.lock().map_err(|err| {
            TensorLoadError::integrity(format!("shard cache lock poisoned: {err}"))
        })?;
        let cached = shards
            .entry(shard_path)
            .or_insert_with(|| Arc::clone(&file));
        tracing::trace!(
            operation = "safetensors_shard_cache_insert",
            inserted = Arc::ptr_eq(cached, &file),
            "safetensors shard cache insert complete"
        );
        Ok(Arc::clone(cached))
    }
}

#[cfg(test)]
mod tests {
    use super::{SafeTensorArchive, SafeTensorFile, mmap_materialization_test_hook};
    use llm_test_support::safetensors::tiny_safetensors_f32;
    use std::{
        sync::{
            Arc, Barrier, Condvar, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    #[test]
    fn archive_from_owned_bytes_reuses_boxed_slice_storage() {
        let bytes = tiny_safetensors_f32("linear.weight", &[1], &[1.0]).into_boxed_slice();
        let ptr = bytes.as_ptr();
        let len = bytes.len();

        let archive = SafeTensorArchive::from_owned_bytes(bytes).expect("archive loads");

        assert_eq!(archive.bytes.as_ref().as_ptr(), ptr);
        assert_eq!(archive.bytes.len(), len);
        assert_eq!(
            archive
                .f32_tensor("linear.weight")
                .expect("tensor still decodes"),
            vec![1.0]
        );
    }

    #[test]
    fn archive_from_bytes_copies_borrowed_storage_and_decodes() {
        let bytes = tiny_safetensors_f32("linear.weight", &[1], &[1.0]);
        let ptr = bytes.as_ptr();
        let len = bytes.len();

        let archive = SafeTensorArchive::from_bytes(bytes.as_slice()).expect("archive loads");

        assert_ne!(archive.bytes.as_ref().as_ptr(), ptr);
        assert_eq!(archive.bytes.len(), len);
        assert_eq!(
            archive
                .f32_tensor("linear.weight")
                .expect("tensor still decodes"),
            vec![1.0]
        );
    }

    #[test]
    fn archive_from_owned_bytes_rejects_invalid_safetensors() {
        let err = SafeTensorArchive::from_owned_bytes(Box::from(*b"not-safetensors"))
            .expect_err("invalid safetensors are rejected");

        assert_eq!(err.code(), "model_integrity_failed");
        assert!(err.message().contains("invalid safetensors"));
    }

    #[test]
    fn concurrent_first_materialize_maps_file_once() {
        let path = std::env::temp_dir().join(format!(
            "llm-backend-concurrent-materialize-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, tiny_safetensors_f32("linear.weight", &[1], &[1.0]))
            .expect("write fixture");
        let file = Arc::new(SafeTensorFile::open(&path).expect("open tensor file"));
        let attempts = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new((Mutex::new(0_usize), Condvar::new()));
        let hook_path = path.clone();
        let hook_attempts = Arc::clone(&attempts);
        let hook_entered = Arc::clone(&entered);
        let _hook_guard = mmap_materialization_test_hook::set(Arc::new(move |source_path| {
            if source_path != Some(hook_path.as_path()) {
                return;
            }
            hook_attempts.fetch_add(1, Ordering::SeqCst);
            let (lock, condvar) = &*hook_entered;
            let mut count = lock.lock().expect("hook count lock");
            *count += 1;
            condvar.notify_all();
            if *count == 1 {
                let (_count, _timeout) = condvar
                    .wait_timeout_while(count, Duration::from_millis(500), |count| *count < 2)
                    .expect("hook count condvar");
            }
        }));
        let start = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let file = Arc::clone(&file);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    file.materialize().expect("materialize tensor file");
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for handle in handles {
            handle.join().expect("materialize thread joins");
        }

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "concurrent first access must perform exactly one mmap materialization"
        );
        std::fs::remove_file(path).ok();
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

fn u64_from_usize(value: usize, message: &str) -> Result<u64, TensorLoadError> {
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

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn integrity(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    pub(crate) fn missing(message: impl Into<String>) -> Self {
        Self {
            code: "model_artifact_missing",
            message: message.into(),
        }
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }

    pub(crate) fn cancelled() -> Self {
        Self {
            code: "request_cancelled",
            message: "request cancelled".to_owned(),
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.code == "request_cancelled"
    }
}
