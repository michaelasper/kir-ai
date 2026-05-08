use std::fmt;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedySampler;

impl GreedySampler {
    pub fn sample(&self, logits: &[f32]) -> Result<usize, SamplerError> {
        select_argmax(logits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopPSampler {
    pub temperature: f32,
    pub top_p: f32,
}

impl TopPSampler {
    pub fn sample(&self, logits: &[f32], draw: f32) -> Result<usize, SamplerError> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature);
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(SamplerError::InvalidTopP);
        }
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(SamplerError::InvalidDraw);
        }
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let mut scaled = Vec::with_capacity(logits.len());
        let mut max_scaled = f32::NEG_INFINITY;
        for (index, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                return Err(SamplerError::NonFiniteLogit { index });
            }
            let value = logit / self.temperature;
            max_scaled = max_scaled.max(value);
            scaled.push((index, value));
        }

        let mut probabilities = Vec::with_capacity(scaled.len());
        let mut sum = 0.0;
        for (index, value) in scaled {
            let probability = (value - max_scaled).exp();
            sum += probability;
            probabilities.push((index, probability));
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }
        for (_, probability) in &mut probabilities {
            *probability /= sum;
        }
        probabilities.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut nucleus = Vec::new();
        let mut nucleus_total = 0.0;
        for (index, probability) in probabilities {
            nucleus_total += probability;
            nucleus.push((index, probability));
            if nucleus_total >= self.top_p {
                break;
            }
        }
        if nucleus.is_empty() || !nucleus_total.is_finite() || nucleus_total <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }

        let threshold = draw * nucleus_total;
        let mut cumulative = 0.0;
        for (index, probability) in &nucleus {
            cumulative += *probability;
            if threshold <= cumulative {
                return Ok(*index);
            }
        }
        Ok(nucleus
            .last()
            .map(|(index, _)| *index)
            .expect("nucleus is not empty"))
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
    InvalidTemperature,
    InvalidTopP,
    InvalidDraw,
    InvalidDistribution,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(formatter, "sampler requires at least one logit"),
            Self::NonFiniteLogit { index } => {
                write!(formatter, "sampler logit at index {index} is not finite")
            }
            Self::InvalidTemperature => {
                write!(formatter, "sampler temperature must be finite and positive")
            }
            Self::InvalidTopP => write!(formatter, "sampler top_p must be in (0, 1]"),
            Self::InvalidDraw => {
                write!(formatter, "sampler draw must be finite and in [0, 1)")
            }
            Self::InvalidDistribution => {
                write!(
                    formatter,
                    "sampler produced an invalid probability distribution"
                )
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

    #[test]
    fn top_p_sampler_selects_within_nucleus_from_draw() {
        let sampler = TopPSampler {
            temperature: 1.0,
            top_p: 0.9,
        };

        assert_eq!(
            sampler
                .sample(&[2.0, 1.0, 0.0], 0.0)
                .expect("low draw samples top token"),
            0
        );
        assert_eq!(
            sampler
                .sample(&[2.0, 1.0, 0.0], 0.8)
                .expect("high draw samples second nucleus token"),
            1
        );
    }

    #[test]
    fn top_p_sampler_keeps_at_least_one_token() {
        let sampler = TopPSampler {
            temperature: 1.0,
            top_p: 0.5,
        };

        let token = sampler
            .sample(&[2.0, 1.0, 0.0], 0.99)
            .expect("sample succeeds");

        assert_eq!(token, 0);
    }

    #[test]
    fn top_p_sampler_rejects_invalid_controls() {
        let err = TopPSampler {
            temperature: 0.0,
            top_p: 1.0,
        }
        .sample(&[1.0], 0.0)
        .expect_err("zero temperature fails");
        assert_eq!(err, SamplerError::InvalidTemperature);

        let err = TopPSampler {
            temperature: 1.0,
            top_p: 1.5,
        }
        .sample(&[1.0], 0.0)
        .expect_err("top_p over one fails");
        assert_eq!(err, SamplerError::InvalidTopP);

        let err = TopPSampler {
            temperature: 1.0,
            top_p: 1.0,
        }
        .sample(&[1.0], 1.0)
        .expect_err("draw outside half-open interval fails");
        assert_eq!(err, SamplerError::InvalidDraw);
    }
}
