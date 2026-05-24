use super::super::math::bf16_bits_to_f32;
use super::TensorLoadError;

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
    Ok(row_bytes
        .chunks_exact(2)
        .zip(input)
        .map(|(chunk, value)| bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])) * value)
        .sum())
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
}
