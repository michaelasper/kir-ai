use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use llm_sampler::{SamplerError, TopPSampler, TopPSamplerScratch};

struct BenchCase {
    name: &'static str,
    vocab: usize,
    temperature: f32,
    top_p: f32,
    draw: f32,
    legacy_iterations: usize,
    current_iterations: usize,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "top_p_32k",
        vocab: 32_000,
        temperature: 0.8,
        top_p: 0.9,
        draw: 0.37,
        legacy_iterations: 10,
        current_iterations: 100,
    },
    BenchCase {
        name: "top_p_128k",
        vocab: 128_000,
        temperature: 0.8,
        top_p: 0.9,
        draw: 0.37,
        legacy_iterations: 5,
        current_iterations: 50,
    },
];

fn main() {
    println!("top_p: legacy full sort vs current partial nucleus selection");
    println!(
        "{:<12} {:<18} {:>10} {:>6} {:>8} {:>12} {:>12}",
        "case", "path", "vocab", "top_p", "iters", "total_ms", "ns/iter"
    );

    for case in CASES {
        let logits = make_logits(case.vocab);
        let sampler = TopPSampler {
            temperature: case.temperature,
            top_p: case.top_p,
        };
        let expected = legacy_full_sort_top_p(&sampler, &logits, case.draw)
            .expect("legacy top-p sample succeeds");
        let actual = sampler
            .sample_with_scratch(&logits, case.draw, &mut TopPSamplerScratch::new())
            .expect("current top-p sample succeeds");
        assert_eq!(actual, expected);

        let legacy_elapsed = run_legacy_case(case, &sampler, &logits);
        print_result(
            case,
            "legacy_full_sort",
            case.legacy_iterations,
            legacy_elapsed,
        );

        let current_elapsed = run_current_case(case, &sampler, &logits);
        print_result(
            case,
            "partial_selection",
            case.current_iterations,
            current_elapsed,
        );
    }
}

fn run_legacy_case(case: &BenchCase, sampler: &TopPSampler, logits: &[f32]) -> Duration {
    let mut checksum = 0_usize;
    for _ in 0..2 {
        checksum ^=
            legacy_full_sort_top_p(black_box(sampler), black_box(logits), black_box(case.draw))
                .expect("legacy warmup succeeds");
    }

    let started = Instant::now();
    for _ in 0..case.legacy_iterations {
        checksum ^=
            legacy_full_sort_top_p(black_box(sampler), black_box(logits), black_box(case.draw))
                .expect("legacy top-p succeeds");
    }
    let elapsed = started.elapsed();
    black_box(checksum);
    elapsed
}

fn run_current_case(case: &BenchCase, sampler: &TopPSampler, logits: &[f32]) -> Duration {
    let mut scratch = TopPSamplerScratch::new();
    let mut checksum = 0_usize;
    for _ in 0..2 {
        checksum ^= sampler
            .sample_with_scratch(black_box(logits), black_box(case.draw), &mut scratch)
            .expect("current warmup succeeds");
    }

    let started = Instant::now();
    for _ in 0..case.current_iterations {
        checksum ^= sampler
            .sample_with_scratch(black_box(logits), black_box(case.draw), &mut scratch)
            .expect("current top-p succeeds");
    }
    let elapsed = started.elapsed();
    black_box(checksum);
    elapsed
}

fn print_result(case: &BenchCase, path: &str, iterations: usize, elapsed: Duration) {
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{:<12} {:<18} {:>10} {:>6.2} {:>8} {:>12.3} {:>12.1}",
        case.name, path, case.vocab, case.top_p, iterations, total_ms, ns_per_iter
    );
}

fn legacy_full_sort_top_p(
    sampler: &TopPSampler,
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

fn make_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|index| {
            let mixed = (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let jitter = (mixed % 4096) as f32 / 2048.0;
            let tail_bias = -((index % 2048) as f32) / 384.0;
            jitter + tail_bias
        })
        .collect()
}
