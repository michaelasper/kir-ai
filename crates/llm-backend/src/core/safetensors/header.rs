use super::{TensorLoadError, usize_from_u64};

const MAX_SAFETENSORS_HEADER_LEN: u64 = 64 * 1024 * 1024;

pub(super) fn read_header_prefix(
    bytes: &[u8],
    file_len: u64,
) -> Result<(u64, usize), TensorLoadError> {
    let prefix = bytes
        .get(0..8)
        .ok_or_else(|| TensorLoadError::integrity("safetensors file is missing header prefix"))?;
    let header_len = validate_header_len(
        u64::from_le_bytes(
            prefix
                .try_into()
                .map_err(|_| TensorLoadError::integrity("header prefix is not 8 bytes"))?,
        ),
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

pub(super) fn validate_header_len(header_len: u64, file_len: u64) -> Result<u64, TensorLoadError> {
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

pub(super) fn parse_shape(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_header_prefix_returns_length_and_header_end_for_valid_prefix() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(b"test");

        let prefix = read_header_prefix(&bytes, bytes.len() as u64).expect("valid prefix");

        assert_eq!(prefix, (4, 12));
    }

    #[test]
    fn read_header_prefix_rejects_missing_prefix() {
        let err = read_header_prefix(&[1, 2, 3], 3).expect_err("missing prefix fails");

        assert_eq!(err.message(), "safetensors file is missing header prefix");
    }

    #[test]
    fn validate_header_len_rejects_lengths_past_file_end() {
        let err = validate_header_len(8, 12).expect_err("header overruns file");

        assert_eq!(
            err.message(),
            "safetensors header length exceeds file length"
        );
    }

    #[test]
    fn parse_shape_returns_usize_dimensions_for_integer_array() {
        let value = serde_json::json!([2, 3, 5]);

        let shape = parse_shape("tensor.weight", Some(&value)).expect("shape parses");

        assert_eq!(shape, vec![2, 3, 5]);
    }

    #[test]
    fn parse_shape_rejects_non_integer_dimensions() {
        let value = serde_json::json!([2, "3"]);

        let err = parse_shape("tensor.weight", Some(&value)).expect_err("shape fails");

        assert_eq!(
            err.message(),
            "tensor `tensor.weight` shape must contain integers"
        );
    }
}
