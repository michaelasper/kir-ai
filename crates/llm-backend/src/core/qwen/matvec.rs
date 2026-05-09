use super::super::NativeMatvecBackend;
use super::super::math::MathError;

#[cfg(test)]
use super::super::CpuNativeMatvecBackend;

pub(super) async fn rms_norm_f32_with_matvec(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; input.len()];
    rms_norm_f32_with_matvec_in_place(input, weight, eps, matvec, &mut output).await?;
    Ok(output)
}

pub(super) async fn rms_norm_f32_with_matvec_in_place(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.len() != weight.len() {
        return Err(MathError::InvalidShape(
            "input and weight must have the same length".to_owned(),
        ));
    }
    let qwen_weight = weight.iter().map(|value| value - 1.0).collect::<Vec<_>>();
    matvec.rms_norm_one_centered_f32_in_place(input, &qwen_weight, eps, output).await
}

pub(super) async fn l2_normalize_f32_with_matvec(
    input: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut qwen_weight = Vec::new();
    let mut output = vec![0.0; input.len()];
    l2_normalize_f32_with_matvec_and_weight_scratch(input, eps, matvec, &mut qwen_weight, &mut output).await?;
    Ok(output)
}

pub(super) async fn l2_normalize_f32_with_matvec_and_weight_scratch(
    input: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
    qwen_weight: &mut Vec<f32>,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.is_empty() {
        qwen_weight.clear();
        return Ok(());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "l2 norm epsilon must be non-negative".to_owned(),
        ));
    }
    let weight_scale = (input.len() as f32).sqrt().recip();
    qwen_weight.clear();
    qwen_weight.resize(input.len(), weight_scale - 1.0);
    matvec.rms_norm_one_centered_f32_in_place(input, qwen_weight, eps / input.len() as f32, output).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn l2_normalize_f32_with_matvec_reuses_weight_scratch() {
        let mut qwen_weight = Vec::with_capacity(8);
        let mut output = vec![0.0; 2];

        l2_normalize_f32_with_matvec_and_weight_scratch(
            &[3.0, 4.0],
            1e-6,
            &CpuNativeMatvecBackend,
            &mut qwen_weight,
            &mut output,
        )
        .await
        .expect("l2 normalize succeeds");

        assert!((output[0] - 0.6).abs() < 1e-5);
        assert!((output[1] - 0.8).abs() < 1e-5);
        assert_eq!(qwen_weight, vec![2.0_f32.sqrt().recip() - 1.0; 2]);
        assert_eq!(qwen_weight.capacity(), 8);
    }
}
