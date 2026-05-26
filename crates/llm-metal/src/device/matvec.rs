use super::command::finish_command_buffer_async;
use super::{Bf16MatrixBuffer, MetalDevice, MetalError};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

fn bf16_matrix_byte_len(matrix: &[u16], rows: usize, cols: usize) -> Result<usize, MetalError> {
    let expected_matrix_len = rows
        .checked_mul(cols)
        .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
    if matrix.len() != expected_matrix_len {
        return Err(MetalError::InvalidShape(format!(
            "matrix length {} does not match rows {rows} * cols {cols}",
            matrix.len()
        )));
    }
    Ok(std::mem::size_of_val(matrix))
}

impl MetalDevice {
    pub async fn matvec_f32(
        &self,
        matrix: &[f32],
        rows: usize,
        cols: usize,
        vector: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        if matrix.len() != expected_matrix_len {
            return Err(MetalError::InvalidShape(format!(
                "matrix length {} does not match rows {rows} * cols {cols}",
                matrix.len()
            )));
        }
        if vector.len() != cols {
            return Err(MetalError::InvalidShape(format!(
                "vector length {} does not match cols {cols}",
                vector.len()
            )));
        }
        if output.len() < rows {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than rows {rows}",
                output.len()
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        if cols == 0 {
            output[..rows].fill(0.0);
            return Ok(());
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let matrix_byte_len = std::mem::size_of_val(matrix) as u64;
        let vector_byte_len = std::mem::size_of_val(vector) as u64;
        let output_byte_len = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let matrix_buffer = self.take_scratch_f32_buffer(matrix);
        let vector_buffer = self.take_scratch_f32_buffer(vector);
        let output_buffer = self.take_scratch_buffer(output_byte_len);

        let command_buffer = self.matvec_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matvec_f32.pipeline);
        encoder.set_buffer(0, Some(&matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .matvec_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer_async(&[], command_buffer, "matvec_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per requested matrix row.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, rows);
            output[..rows].copy_from_slice(values);
        };
        self.return_scratch_buffer(matrix_byte_len, matrix_buffer);
        self.return_scratch_buffer(vector_byte_len, vector_buffer);
        self.return_scratch_buffer(output_byte_len, output_buffer);
        Ok(())
    }

    pub async fn matvec_bf16_f32(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
        vector: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let byte_len = bf16_matrix_byte_len(matrix, rows, cols)?;
        let mut matrix_buffer = Bf16MatrixBuffer {
            buffer: (byte_len != 0).then(|| self.take_scratch_bf16_buffer(matrix)),
            rows,
            columns: cols,
            byte_len,
        };

        let result = self
            .matvec_bf16_f32_buffered(&matrix_buffer, vector, output)
            .await;
        if let Some(buffer) = matrix_buffer.buffer.take() {
            self.return_scratch_buffer(byte_len as u64, buffer);
        }
        result
    }

    pub fn new_bf16_matrix_buffer(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<Bf16MatrixBuffer, MetalError> {
        let byte_len = bf16_matrix_byte_len(matrix, rows, cols)?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.device.new_buffer_with_data(
                matrix.as_ptr().cast::<c_void>(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(Bf16MatrixBuffer {
            buffer,
            rows,
            columns: cols,
            byte_len,
        })
    }

    pub async fn matvec_bf16_f32_buffered(
        &self,
        matrix: &Bf16MatrixBuffer,
        vector: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let rows = matrix.rows;
        let cols = matrix.columns;
        if vector.len() != cols {
            return Err(MetalError::InvalidShape(format!(
                "vector length {} does not match cols {cols}",
                vector.len()
            )));
        }
        if output.len() < rows {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than rows {rows}",
                output.len()
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        if cols == 0 {
            output[..rows].fill(0.0);
            return Ok(());
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let vector_byte_len = std::mem::size_of_val(vector) as u64;
        let output_byte_len = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let Some(matrix_buffer) = matrix.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty BF16 matvec requires a matrix buffer".to_owned(),
            ));
        };
        let vector_buffer = self.take_scratch_f32_buffer(vector);
        let output_buffer = self.take_scratch_buffer(output_byte_len);

        let command_buffer = self.matvec_bf16_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matvec_bf16_f32.pipeline);
        encoder.set_buffer(0, Some(matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .matvec_bf16_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer_async(&[], command_buffer, "matvec_bf16_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per requested matrix row.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, rows);
            output[..rows].copy_from_slice(values);
        };
        self.return_scratch_buffer(vector_byte_len, vector_buffer);
        self.return_scratch_buffer(output_byte_len, output_buffer);
        Ok(())
    }

    pub async fn batched_matvec_bf16_f32(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
        vectors: &[f32],
        vector_count: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let byte_len = bf16_matrix_byte_len(matrix, rows, cols)?;
        let mut matrix_buffer = Bf16MatrixBuffer {
            buffer: (byte_len != 0).then(|| self.take_scratch_bf16_buffer(matrix)),
            rows,
            columns: cols,
            byte_len,
        };

        let result = self
            .batched_matvec_bf16_f32_buffered(&matrix_buffer, vectors, vector_count, output)
            .await;
        if let Some(buffer) = matrix_buffer.buffer.take() {
            self.return_scratch_buffer(byte_len as u64, buffer);
        }
        result
    }

    pub async fn batched_matvec_bf16_f32_buffered(
        &self,
        matrix: &Bf16MatrixBuffer,
        vectors: &[f32],
        vector_count: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let rows = matrix.rows;
        let cols = matrix.columns;
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        debug_assert_eq!(
            matrix.byte_len / std::mem::size_of::<u16>(),
            expected_matrix_len
        );
        let expected_vectors_len = vector_count.checked_mul(cols).ok_or_else(|| {
            MetalError::InvalidShape("batched vector shape overflows usize".to_owned())
        })?;
        if vectors.len() != expected_vectors_len {
            return Err(MetalError::InvalidShape(format!(
                "batched vector length {} does not match vector_count {vector_count} * cols {cols}",
                vectors.len()
            )));
        }
        let output_len = vector_count.checked_mul(rows).ok_or_else(|| {
            MetalError::InvalidShape("batched output shape overflows usize".to_owned())
        })?;
        if output.len() < output_len {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than expected {output_len}",
                output.len()
            )));
        }
        if rows == 0 || vector_count == 0 {
            return Ok(());
        }
        if cols == 0 {
            output[..output_len].fill(0.0);
            return Ok(());
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let vector_count_u32 = u32::try_from(vector_count).map_err(|err| {
            MetalError::InvalidShape(format!("vector count does not fit u32: {err}"))
        })?;
        let vector_byte_len = std::mem::size_of_val(vectors) as u64;
        let output_byte_len = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let Some(matrix_buffer) = matrix.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty batched BF16 matvec requires a matrix buffer".to_owned(),
            ));
        };
        let vector_buffer = self.take_scratch_f32_buffer(vectors);
        let output_buffer = self.take_scratch_buffer(output_byte_len);

        let command_buffer = self.batched_matvec_bf16_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_matvec_bf16_f32.pipeline);
        encoder.set_buffer(0, Some(matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&vector_count_u32) as u64,
            (&vector_count_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(5, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: vector_count as u64,
            depth: 1,
        };
        let group_width = self
            .batched_matvec_bf16_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer_async(&[], command_buffer, "batched_matvec_bf16_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing vector_count * rows f32 values in input-major order.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, output_len);
            output[..output_len].copy_from_slice(values);
        };
        self.return_scratch_buffer(vector_byte_len, vector_buffer);
        self.return_scratch_buffer(output_byte_len, output_buffer);
        Ok(())
    }
}
