use super::super::math::bf16_bits_to_f32;
use super::TensorLoadError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bf16DotKernel {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

pub(super) fn bf16_row_byte_len(columns: usize, context: &str) -> Result<usize, TensorLoadError> {
    columns
        .checked_mul(2)
        .ok_or_else(|| TensorLoadError::integrity(format!("{context} row byte length overflow")))
}

pub(super) fn bf16_row_bytes<'a>(
    bytes: &'a [u8],
    row_offset: usize,
    row_byte_len: usize,
    context: &str,
) -> Result<&'a [u8], TensorLoadError> {
    let start = row_offset
        .checked_mul(row_byte_len)
        .ok_or_else(|| TensorLoadError::integrity(format!("{context} row offset overflow")))?;
    let end = start
        .checked_add(row_byte_len)
        .ok_or_else(|| TensorLoadError::integrity(format!("{context} row range overflow")))?;
    bytes
        .get(start..end)
        .ok_or_else(|| TensorLoadError::integrity(format!("{context} BF16 row range is invalid")))
}

pub(super) fn bf16_dot_f32(row_bytes: &[u8], input: &[f32]) -> Result<f32, TensorLoadError> {
    let expected_byte_len = input
        .len()
        .checked_mul(2)
        .ok_or_else(|| TensorLoadError::integrity("BF16 dot input byte length overflow"))?;
    if row_bytes.len() != expected_byte_len {
        return Err(TensorLoadError::integrity(format!(
            "BF16 row byte length {} does not match input byte length {expected_byte_len}",
            row_bytes.len()
        )));
    }
    Ok(match bf16_dot_f32_kernel() {
        Bf16DotKernel::Scalar => bf16_dot_f32_scalar(row_bytes, input),
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Bf16DotKernel::Neon => {
            // SAFETY: kernel selection only returns Neon when the target supports NEON,
            // and the byte/input lengths were validated above.
            unsafe { bf16_dot_f32_neon(row_bytes, input) }
        }
    })
}

fn bf16_dot_f32_kernel() -> Bf16DotKernel {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if aarch64_neon_available() {
            return Bf16DotKernel::Neon;
        }
    }
    Bf16DotKernel::Scalar
}

fn bf16_dot_f32_scalar(row_bytes: &[u8], input: &[f32]) -> f32 {
    row_bytes
        .chunks_exact(2)
        .zip(input)
        .map(|(chunk, value)| bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])) * value)
        .sum()
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn aarch64_neon_available() -> bool {
    #[cfg(target_feature = "neon")]
    {
        true
    }
    #[cfg(not(target_feature = "neon"))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn bf16_dot_f32_neon(row_bytes: &[u8], input: &[f32]) -> f32 {
    use std::arch::aarch64::{
        vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1_u8, vld1q_f32, vmovl_u16, vreinterpret_u16_u8,
        vreinterpretq_f32_u32, vshlq_n_u32,
    };

    let vector_len = input.len() / 4 * 4;
    let mut index = 0;
    let mut acc = vdupq_n_f32(0.0);

    while index < vector_len {
        // SAFETY: `bf16_dot_f32` validates that `row_bytes` contains exactly
        // two bytes per input element. This loop only reads full 4-element
        // chunks, so the 8-byte BF16 load and 4-lane F32 load are in bounds.
        let (weights, values) = unsafe {
            // Loading as bytes avoids imposing a `u16` alignment requirement
            // on safetensors row storage.
            let bytes = vld1_u8(row_bytes.as_ptr().add(index * 2));
            let bf16_lanes = vreinterpret_u16_u8(bytes);
            let f32_bits = vshlq_n_u32(vmovl_u16(bf16_lanes), 16);
            (
                vreinterpretq_f32_u32(f32_bits),
                vld1q_f32(input.as_ptr().add(index)),
            )
        };
        acc = vfmaq_f32(acc, weights, values);
        index += 4;
    }

    let mut sum = vaddvq_f32(acc);
    while index < input.len() {
        let byte_index = index * 2;
        // SAFETY: the caller validated two bytes per input element, and this
        // loop is bounded by `input.len()`.
        let product = unsafe {
            let bits = u16::from_le_bytes([
                *row_bytes.get_unchecked(byte_index),
                *row_bytes.get_unchecked(byte_index + 1),
            ]);
            bf16_bits_to_f32(bits) * *input.get_unchecked(index)
        };
        sum += product;
        index += 1;
    }
    sum
}

pub(super) fn bf16_bytes_to_f32_into(
    bytes: &[u8],
    output: &mut Vec<f32>,
) -> Result<(), TensorLoadError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(TensorLoadError::integrity(
            "BF16 byte length must be divisible by 2",
        ));
    }
    output.clear();
    output.reserve(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        output.push(bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(())
}

pub(super) fn bf16_bytes_to_bits(bytes: &[u8]) -> Result<Vec<u16>, TensorLoadError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(TensorLoadError::integrity(
            "BF16 byte length must be divisible by 2",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bytes_to_f32_into_decodes_values_into_existing_buffer() {
        let bytes = bf16_bytes(&[1.0, 2.5, -4.0]);
        let mut output = Vec::with_capacity(8);
        output.extend_from_slice(&[99.0, 100.0]);
        let original_ptr = output.as_ptr();

        bf16_bytes_to_f32_into(&bytes, &mut output).expect("BF16 bytes decode");

        assert_eq!(output, vec![1.0, 2.5, -4.0]);
        assert_eq!(output.as_ptr(), original_ptr);
    }

    #[test]
    fn bf16_bytes_to_f32_into_rejects_odd_byte_count() {
        let err = bf16_bytes_to_f32_into(&[0_u8], &mut Vec::new()).expect_err("odd bytes fail");

        assert_eq!(err.message(), "BF16 byte length must be divisible by 2");
    }

    #[test]
    fn bf16_dot_f32_computes_dot_product_without_full_row_decode() {
        let bytes = bf16_bytes(&[1.0, 2.0, 3.0]);

        let dot = bf16_dot_f32(&bytes, &[4.0, 5.0, 6.0]).expect("dot computes");

        assert_eq!(dot, 32.0);
    }

    #[test]
    fn bf16_dot_f32_matches_scalar_kernel_across_tail_lengths() {
        for len in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33] {
            let weights = patterned_values(len, 0.25);
            let input = patterned_values(len, -0.5);
            let bytes = bf16_bytes(&weights);

            let dot = bf16_dot_f32(&bytes, &input).expect("dot computes");
            let scalar = bf16_dot_f32_scalar(&bytes, &input);

            assert_close(dot, scalar, len);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn bf16_dot_f32_selects_neon_kernel_when_available() {
        if std::arch::is_aarch64_feature_detected!("neon") {
            assert_eq!(bf16_dot_f32_kernel(), Bf16DotKernel::Neon);
        }
    }

    #[test]
    fn bf16_dot_f32_rejects_input_width_mismatch() {
        let bytes = bf16_bytes(&[1.0, 2.0]);

        let err = bf16_dot_f32(&bytes, &[1.0]).expect_err("mismatch fails");

        assert_eq!(
            err.message(),
            "BF16 row byte length 4 does not match input byte length 2"
        );
    }

    #[test]
    fn bf16_row_bytes_returns_requested_row_slice() {
        let bytes = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

        let row = bf16_row_bytes(&bytes, 1, 4, "test row").expect("row slice");

        assert_eq!(row, &bytes[4..8]);
    }

    #[test]
    fn bf16_bytes_to_bits_preserves_raw_little_endian_values() {
        let bytes = [0x34, 0x12, 0x78, 0x56];

        let bits = bf16_bytes_to_bits(&bytes).expect("bits decode");

        assert_eq!(bits, vec![0x1234, 0x5678]);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
        }
        bytes
    }

    fn f32_to_bf16_bits(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn patterned_values(len: usize, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let signed = (index as i32 % 9) - 4;
                (signed as f32 * 0.375) + offset
            })
            .collect()
    }

    fn assert_close(actual: f32, expected: f32, len: usize) {
        let delta = (actual - expected).abs();
        let tolerance = 1e-4_f32.max(expected.abs() * 1e-5);
        assert!(
            delta <= tolerance,
            "length {len}: actual {actual} expected {expected} delta {delta} tolerance {tolerance}"
        );
    }
}
