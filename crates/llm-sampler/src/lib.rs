use std::fmt;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedySampler;

impl GreedySampler {
    pub fn sample(&self, logits: &[f32]) -> Result<usize, SamplerError> {
        select_argmax(logits)
    }
}

pub fn select_argmax(logits: &[f32]) -> Result<usize, SamplerError> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    let mut best_index = 0;
    let mut best_logit = logits[0];
    if !best_logit.is_finite() {
        return Err(SamplerError::NonFiniteLogit { index: 0 });
    }
    for (index, logit) in logits.iter().copied().enumerate().skip(1) {
        if !logit.is_finite() {
            return Err(SamplerError::NonFiniteLogit { index });
        }
        if logit > best_logit {
            best_index = index;
            best_logit = logit;
        }
    }
    Ok(best_index)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerError {
    EmptyLogits,
    NonFiniteLogit { index: usize },
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(formatter, "sampler requires at least one logit"),
            Self::NonFiniteLogit { index } => {
                write!(formatter, "sampler logit at index {index} is not finite")
            }
        }
    }
}

impl std::error::Error for SamplerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampler_selects_highest_logit() {
        let sampler = GreedySampler;

        let token = sampler
            .sample(&[-1.0, 0.25, 3.5, 3.0])
            .expect("sample succeeds");

        assert_eq!(token, 2);
    }

    #[test]
    fn argmax_keeps_first_token_on_ties() {
        let token = select_argmax(&[1.0, 2.0, 2.0]).expect("sample succeeds");

        assert_eq!(token, 1);
    }

    #[test]
    fn sampler_rejects_empty_logits() {
        let err = select_argmax(&[]).expect_err("empty logits fail");

        assert_eq!(err, SamplerError::EmptyLogits);
    }

    #[test]
    fn sampler_rejects_non_finite_logits() {
        let err = select_argmax(&[0.0, f32::NAN]).expect_err("nan fails");

        assert_eq!(err, SamplerError::NonFiniteLogit { index: 1 });
    }
}
