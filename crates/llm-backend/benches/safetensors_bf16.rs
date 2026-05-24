use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use llm_backend::native::{
    NativeBatchedMatvecOutput, SafeTensorFile, SafeTensorShardStore, TensorLoadError,
};

const TENSOR: &str = "embed.weight";
const ROWS: usize = 256;
const COLUMNS: usize = 512;
const ROW_ITERATIONS: usize = 10_000;
const MATVEC_ITERATIONS: usize = 200;
const MATVEC_CHUNK_ROWS: usize = 64;
const PREFILL_COPY_CHUNKS: &[usize] = &[1, 4, 16, 64, 128];

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: this allocator delegates to the process System allocator with
        // the layout supplied by Rust's allocation machinery.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are forwarded unchanged from the caller to
        // the same System allocator used for allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: `ptr`, `layout`, and `new_size` are forwarded unchanged from
        // Rust's allocation machinery to the System allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

struct BenchResult {
    elapsed: Duration,
    allocations: u64,
    allocated_bytes: u64,
}

fn main() -> Result<(), TensorLoadError> {
    let root = temp_snapshot_dir("safetensors-bf16-bench");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).map_err(|err| {
        TensorLoadError::integrity(format!("could not create benchmark snapshot: {err}"))
    })?;
    std::fs::write(
        root.join("model.safetensors"),
        tiny_safetensors_bf16(TENSOR, &[ROWS, COLUMNS], &bench_values()),
    )
    .map_err(|err| TensorLoadError::integrity(format!("could not write benchmark shard: {err}")))?;

    let file = SafeTensorFile::open(root.join("model.safetensors"))?;
    file.materialize()?;
    let store = SafeTensorShardStore::open(&root)?;
    store.materialize_shard_for_tensor(TENSOR)?;
    store.bf16_matvec_rows_f32_in_place(
        TENSOR,
        &bench_input(),
        MATVEC_CHUNK_ROWS,
        &mut vec![0.0; ROWS],
    )?;

    println!("safetensors_bf16: legacy mmap copy/decode vs borrowed-byte hot paths");
    println!(
        "{:<18} {:<24} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "case", "path", "iters", "total_ms", "ns/iter", "alloc/iter", "bytes/iter"
    );

    let row_index = ROWS / 2;
    let row_legacy = measure(ROW_ITERATIONS, || {
        legacy_bf16_row_f32(&file, TENSOR, row_index).map(|values| checksum_f32(&values))
    })?;
    print_result(
        "row_read",
        "legacy_copy_decode",
        ROW_ITERATIONS,
        &row_legacy,
    );

    let row_current = measure(ROW_ITERATIONS, || {
        file.bf16_row_f32(TENSOR, row_index)
            .map(|values| checksum_f32(&values))
    })?;
    print_result("row_read", "borrowed_decode", ROW_ITERATIONS, &row_current);

    let mut row_buffer = Vec::with_capacity(COLUMNS);
    let row_buffered = measure(ROW_ITERATIONS, || {
        file.bf16_row_f32_into(TENSOR, row_index, &mut row_buffer)?;
        Ok(checksum_f32(&row_buffer))
    })?;
    print_result("row_read", "caller_buffer", ROW_ITERATIONS, &row_buffered);

    let input = bench_input();
    let mut legacy_output = vec![0.0; ROWS];
    let matvec_legacy = measure(MATVEC_ITERATIONS, || {
        legacy_bf16_matvec_rows_f32_in_place(
            &file,
            TENSOR,
            &input,
            MATVEC_CHUNK_ROWS,
            &mut legacy_output,
        )?;
        Ok(checksum_f32(&legacy_output))
    })?;
    print_result(
        "matvec_chunks",
        "legacy_chunk_decode",
        MATVEC_ITERATIONS,
        &matvec_legacy,
    );

    let mut current_output = vec![0.0; ROWS];
    let matvec_current = measure(MATVEC_ITERATIONS, || {
        store.bf16_matvec_rows_f32_in_place(
            TENSOR,
            &input,
            MATVEC_CHUNK_ROWS,
            &mut current_output,
        )?;
        Ok(checksum_f32(&current_output))
    })?;
    print_result(
        "matvec_chunks",
        "borrowed_direct_dot",
        MATVEC_ITERATIONS,
        &matvec_current,
    );

    println!();
    println!("batched_matvec_copy_overhead: prefill chunk flatten/split vs flat views");
    println!(
        "{:<18} {:<24} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "case", "path", "iters", "total_ms", "ns/iter", "alloc/iter", "bytes/iter"
    );
    run_prefill_copy_cases()?;

    std::fs::remove_dir_all(root).ok();
    Ok(())
}

fn measure(
    iterations: usize,
    mut run: impl FnMut() -> Result<u64, TensorLoadError>,
) -> Result<BenchResult, TensorLoadError> {
    let mut checksum = 0_u64;
    for _ in 0..3 {
        checksum ^= black_box(run()?);
    }
    reset_allocations();
    let started = Instant::now();
    for _ in 0..iterations {
        checksum ^= black_box(run()?);
    }
    let elapsed = started.elapsed();
    black_box(checksum);
    Ok(BenchResult {
        elapsed,
        allocations: ALLOCATION_COUNT.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    })
}

fn reset_allocations() {
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn print_result(case: &str, path: &str, iterations: usize, result: &BenchResult) {
    let total_ms = result.elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = result.elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    let allocations_per_iter = result.allocations as f64 / iterations as f64;
    let bytes_per_iter = result.allocated_bytes as f64 / iterations as f64;
    println!(
        "{:<18} {:<24} {:>8} {:>12.3} {:>12.1} {:>12.2} {:>12.1}",
        case, path, iterations, total_ms, ns_per_iter, allocations_per_iter, bytes_per_iter
    );
}

fn run_prefill_copy_cases() -> Result<(), TensorLoadError> {
    let max_chunk = PREFILL_COPY_CHUNKS.iter().copied().max().unwrap_or(0);
    let row_inputs = prefill_row_inputs(max_chunk);
    let flat_inputs = flatten_prefill_rows(&row_inputs, max_chunk);
    let output_values = prefill_output_values(max_chunk);

    for &chunk in PREFILL_COPY_CHUNKS {
        let iterations = prefill_copy_iterations(chunk);
        let case = format!("prefill_{chunk}");
        let legacy = measure(iterations, || {
            let mut flattened = Vec::with_capacity(chunk * COLUMNS);
            for input in row_inputs.iter().take(chunk) {
                flattened.extend_from_slice(input);
            }
            let rows = output_values[..chunk * ROWS]
                .chunks_exact(ROWS)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            Ok(checksum_f32(&flattened) ^ checksum_nested_rows(&rows))
        })?;
        print_result(&case, "legacy_flatten_split", iterations, &legacy);

        let output = NativeBatchedMatvecOutput::new(output_values[..chunk * ROWS].to_vec(), ROWS)?;
        let flat_input = &flat_inputs[..chunk * COLUMNS];
        let current = measure(iterations, || {
            Ok(checksum_f32(flat_input) ^ checksum_row_views(output.rows()))
        })?;
        print_result(&case, "flat_input_row_views", iterations, &current);
    }

    Ok(())
}

fn prefill_copy_iterations(chunk: usize) -> usize {
    match chunk {
        0..=4 => 20_000,
        5..=16 => 10_000,
        17..=64 => 3_000,
        _ => 1_000,
    }
}

fn legacy_bf16_row_f32(
    file: &SafeTensorFile,
    name: &str,
    row: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    let metadata = file.tensor_metadata(name)?;
    if metadata.shape.len() != 2 {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{name}` row reader expects rank 2, got rank {}",
            metadata.shape.len()
        )));
    }
    let rows = metadata.shape[0];
    let columns = metadata.shape[1];
    if row >= rows {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{name}` row {row} exceeds row count {rows}"
        )));
    }
    let element_offset = row
        .checked_mul(columns)
        .ok_or_else(|| TensorLoadError::integrity("row offset overflow"))?;
    legacy_bf16_tensor_f32_range(file, name, element_offset, columns)
}

fn legacy_bf16_tensor_f32_range(
    file: &SafeTensorFile,
    name: &str,
    element_offset: usize,
    element_count: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    let metadata = file.tensor_metadata(name)?;
    if metadata.dtype != "BF16" {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{name}` has dtype {}, expected BF16",
            metadata.dtype
        )));
    }
    let byte_offset = u64::try_from(
        element_offset
            .checked_mul(2)
            .ok_or_else(|| TensorLoadError::integrity("BF16 element offset overflow"))?,
    )
    .map_err(|_| TensorLoadError::integrity("BF16 byte offset does not fit in u64"))?;
    let byte_len = element_count
        .checked_mul(2)
        .ok_or_else(|| TensorLoadError::integrity("BF16 element count overflow"))?;
    let bytes = file.tensor_bytes_range(name, byte_offset, byte_len)?;
    legacy_bf16_bytes_to_f32(&bytes)
}

fn legacy_bf16_matvec_rows_f32_in_place(
    file: &SafeTensorFile,
    tensor: &str,
    input: &[f32],
    chunk_rows: usize,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let metadata = file.tensor_metadata(tensor)?;
    if metadata.shape.len() != 2 {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{tensor}` matvec expects rank 2, got rank {}",
            metadata.shape.len()
        )));
    }
    let rows = metadata.shape[0];
    let columns = metadata.shape[1];
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
    for row_start in (0..rows).step_by(chunk_rows) {
        let rows_in_chunk = chunk_rows.min(rows - row_start);
        let element_offset = row_start
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("matvec offset overflow"))?;
        let element_count = rows_in_chunk
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("matvec chunk overflow"))?;
        let weights = legacy_bf16_tensor_f32_range(file, tensor, element_offset, element_count)?;
        for (row_offset, row) in weights.chunks_exact(columns).enumerate() {
            output[row_start + row_offset] = row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
    }
    Ok(())
}

fn legacy_bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, TensorLoadError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(TensorLoadError::integrity(
            "BF16 byte length must be divisible by 2",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn bench_values() -> Vec<f32> {
    (0..ROWS * COLUMNS)
        .map(|index| (index % 37) as f32 / 16.0 - 1.0)
        .collect()
}

fn bench_input() -> Vec<f32> {
    (0..COLUMNS)
        .map(|index| (index % 17) as f32 / 32.0 + 0.25)
        .collect()
}

fn prefill_row_inputs(rows: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..COLUMNS)
                .map(|column| ((row * 31 + column) % 257) as f32 / 128.0 - 1.0)
                .collect()
        })
        .collect()
}

fn flatten_prefill_rows(row_inputs: &[Vec<f32>], rows: usize) -> Vec<f32> {
    let mut flattened = Vec::with_capacity(rows * COLUMNS);
    for input in row_inputs.iter().take(rows) {
        flattened.extend_from_slice(input);
    }
    flattened
}

fn prefill_output_values(rows: usize) -> Vec<f32> {
    (0..rows * ROWS)
        .map(|index| (index % 131) as f32 / 64.0 - 0.5)
        .collect()
}

fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
}

fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let header = serde_json::json!({
        name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, data.len()]
        }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llm-backend-{label}-{}", std::process::id()))
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn checksum_f32(values: &[f32]) -> u64 {
    values.iter().fold(0_u64, |checksum, value| {
        checksum.rotate_left(5) ^ value.to_bits() as u64
    })
}

fn checksum_nested_rows(rows: &[Vec<f32>]) -> u64 {
    rows.iter()
        .fold(0_u64, |checksum, row| checksum ^ checksum_f32(row))
}

fn checksum_row_views<'a>(rows: impl IntoIterator<Item = &'a [f32]>) -> u64 {
    rows.into_iter()
        .fold(0_u64, |checksum, row| checksum ^ checksum_f32(row))
}
