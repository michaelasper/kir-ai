use std::{cmp::Ordering, fmt};

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

#[derive(Debug, Clone, Copy, PartialEq)]
struct RankedProbability {
    index: usize,
    probability: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TopPSamplerScratch {
    ranked_probabilities: Vec<RankedProbability>,
}

impl TopPSamplerScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capacity(&self) -> usize {
        self.ranked_probabilities.capacity()
    }
}

impl TopPSampler {
    pub fn sample(&self, logits: &[f32], draw: f32) -> Result<usize, SamplerError> {
        let mut scratch = TopPSamplerScratch::new();
        self.sample_with_scratch(logits, draw, &mut scratch)
    }

    pub fn sample_with_scratch(
        &self,
        logits: &[f32],
        draw: f32,
        scratch: &mut TopPSamplerScratch,
    ) -> Result<usize, SamplerError> {
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

        scratch.ranked_probabilities.clear();
        scratch.ranked_probabilities.reserve(logits.len());
        let mut max_scaled = f32::NEG_INFINITY;
        for (index, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                return Err(SamplerError::NonFiniteLogit { index });
            }
            let value = logit / self.temperature;
            max_scaled = max_scaled.max(value);
            scratch.ranked_probabilities.push(RankedProbability {
                index,
                probability: value,
            });
        }

        let mut sum = 0.0;
        for entry in &mut scratch.ranked_probabilities {
            entry.probability = (entry.probability - max_scaled).exp();
            sum += entry.probability;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }
        for entry in &mut scratch.ranked_probabilities {
            entry.probability /= sum;
        }

        let (nucleus_len, nucleus_total) =
            select_sorted_nucleus_prefix(&mut scratch.ranked_probabilities, self.top_p)?;

        let threshold = draw * nucleus_total;
        let mut cumulative = 0.0;
        for entry in scratch.ranked_probabilities.iter().take(nucleus_len) {
            cumulative += entry.probability;
            if threshold <= cumulative {
                return Ok(entry.index);
            }
        }
        scratch
            .ranked_probabilities
            .get(nucleus_len - 1)
            .map(|entry| entry.index)
            .ok_or(SamplerError::InvalidDistribution)
    }
}

fn select_sorted_nucleus_prefix(
    entries: &mut [RankedProbability],
    top_p: f32,
) -> Result<(usize, f32), SamplerError> {
    let len = entries.len();
    let mut candidate_len = initial_candidate_len(len);

    loop {
        if candidate_len < len {
            entries.select_nth_unstable_by(candidate_len - 1, ranked_probability_order);
        }
        entries[..candidate_len].sort_by(ranked_probability_order);

        let mut nucleus_total = 0.0;
        for (offset, entry) in entries[..candidate_len].iter().enumerate() {
            nucleus_total += entry.probability;
            if !nucleus_total.is_finite() {
                return Err(SamplerError::InvalidDistribution);
            }
            if nucleus_total >= top_p {
                return Ok((offset + 1, nucleus_total));
            }
        }

        if candidate_len == len {
            if candidate_len == 0 || nucleus_total <= 0.0 {
                return Err(SamplerError::InvalidDistribution);
            }
            return Ok((candidate_len, nucleus_total));
        }

        candidate_len = grow_candidate_len(candidate_len, len);
    }
}

fn initial_candidate_len(len: usize) -> usize {
    len.min(4096)
}

fn grow_candidate_len(current: usize, len: usize) -> usize {
    current.saturating_mul(4).min(len).max(current + 1)
}

fn ranked_probability_order(left: &RankedProbability, right: &RankedProbability) -> Ordering {
    right
        .probability
        .total_cmp(&left.probability)
        .then_with(|| left.index.cmp(&right.index))
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
    fn top_p_sampler_uses_index_order_for_probability_ties() {
        let sampler = TopPSampler {
            temperature: 1.0,
            top_p: 0.6,
        };

        assert_eq!(
            sampler
                .sample(&[2.0, 2.0, 2.0, 0.0], 0.0)
                .expect("low draw samples first tied token"),
            0
        );
        assert_eq!(
            sampler
                .sample(&[2.0, 2.0, 2.0, 0.0], 0.99)
                .expect("high draw samples second tied token"),
            1
        );
    }

    #[test]
    fn top_p_sampler_matches_full_sort_reference_for_representative_draws() {
        for vocab in [8_usize, 257, 4096] {
            let logits = make_test_logits(vocab);
            let mut scratch = TopPSamplerScratch::new();
            for top_p in [0.05_f32, 0.5, 0.9, 1.0] {
                let sampler = TopPSampler {
                    temperature: 0.7,
                    top_p,
                };
                for draw in [0.0_f32, 0.37, 0.999_999] {
                    let expected = legacy_full_sort_top_p(sampler, &logits, draw)
                        .expect("legacy sample succeeds");
                    let actual = sampler
                        .sample_with_scratch(&logits, draw, &mut scratch)
                        .expect("optimized sample succeeds");

                    assert_eq!(actual, expected, "vocab={vocab} top_p={top_p} draw={draw}");
                }
            }
        }
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
    fn top_p_sampler_reuses_scratch_capacity_across_samples() {
        let sampler = TopPSampler {
            temperature: 1.0,
            top_p: 0.9,
        };
        let mut scratch = TopPSamplerScratch::new();

        assert_eq!(
            sampler
                .sample_with_scratch(&[2.0, 1.0, 0.0], 0.0, &mut scratch)
                .expect("first sample succeeds"),
            0
        );
        let capacity = scratch.capacity();
        assert!(capacity >= 3);

        assert_eq!(
            sampler
                .sample_with_scratch(&[2.0, 1.0, 0.0], 0.8, &mut scratch)
                .expect("second sample succeeds"),
            1
        );

        assert_eq!(scratch.capacity(), capacity);
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

    fn legacy_full_sort_top_p(
        sampler: TopPSampler,
        logits: &[f32],
        draw: f32,
    ) -> Result<usize, SamplerError> {
        if !sampler.temperature.is_finite() || sampler.temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature);
        }
        if !sampler.top_p.is_finite() || sampler.top_p <= 0.0 || sampler.top_p > 1.0 {
            return Err(SamplerError::InvalidTopP);
        }
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(SamplerError::InvalidDraw);
        }
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let mut ranked_probabilities = Vec::with_capacity(logits.len());
        let mut max_scaled = f32::NEG_INFINITY;
        for (index, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                return Err(SamplerError::NonFiniteLogit { index });
            }
            let value = logit / sampler.temperature;
            max_scaled = max_scaled.max(value);
            ranked_probabilities.push((index, value));
        }

        let mut sum = 0.0;
        for (_, value) in &mut ranked_probabilities {
            *value = (*value - max_scaled).exp();
            sum += *value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }
        for (_, probability) in &mut ranked_probabilities {
            *probability /= sum;
        }
        ranked_probabilities.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut nucleus_total = 0.0;
        let mut nucleus_len = 0;
        for (_, probability) in &ranked_probabilities {
            nucleus_total += *probability;
            nucleus_len += 1;
            if nucleus_total >= sampler.top_p {
                break;
            }
        }
        if nucleus_len == 0 || !nucleus_total.is_finite() || nucleus_total <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }

        let threshold = draw * nucleus_total;
        let mut cumulative = 0.0;
        for (index, probability) in ranked_probabilities.iter().take(nucleus_len) {
            cumulative += *probability;
            if threshold <= cumulative {
                return Ok(*index);
            }
        }
        ranked_probabilities
            .get(nucleus_len - 1)
            .map(|(index, _)| *index)
            .ok_or(SamplerError::InvalidDistribution)
    }

    fn make_test_logits(vocab: usize) -> Vec<f32> {
        (0..vocab)
            .map(|index| {
                let mixed = (index as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                (mixed % 4096) as f32 / 256.0 - 8.0
            })
            .collect()
    }
}
