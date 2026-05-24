use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use llm_backend::native::{CpuNativeMatvecBackend, MathError, NativeMatvecBackend, TopKWeight};
use tokio::runtime::Builder;

struct BenchCase {
    name: &'static str,
    vocab: usize,
    top_k: usize,
    legacy_iterations: usize,
    current_iterations: usize,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "moe_router_32k_k2",
        vocab: 32_000,
        top_k: 2,
        legacy_iterations: 10,
        current_iterations: 100,
    },
    BenchCase {
        name: "sampler_128k_k8",
        vocab: 128_000,
        top_k: 8,
        legacy_iterations: 5,
        current_iterations: 50,
    },
    BenchCase {
        name: "sampler_152k_k64",
        vocab: 151_936,
        top_k: 64,
        legacy_iterations: 3,
        current_iterations: 20,
    },
    BenchCase {
        name: "sampler_152k_k256",
        vocab: 151_936,
        top_k: 256,
        legacy_iterations: 2,
        current_iterations: 10,
    },
];

fn main() {
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime builds");
    let backend = CpuNativeMatvecBackend;

    println!("top_k_selection: legacy full sort vs current bounded/partial selection");
    println!(
        "{:<22} {:<18} {:>10} {:>6} {:>8} {:>12} {:>12}",
        "case", "path", "vocab", "k", "iters", "total_ms", "ns/iter"
    );

    for case in CASES {
        let logits = make_logits(case.vocab);
        let legacy =
            legacy_full_sort_top_k_f32(&logits, case.top_k).expect("legacy top-k succeeds");
        let current = runtime
            .block_on(backend.softmax_top_k_f32(&logits, case.top_k))
            .expect("current top-k succeeds");
        assert_same_selection(&legacy, &current);

        let legacy_elapsed = run_legacy_case(&logits, case.top_k, case.legacy_iterations);
        print_result(
            case,
            "legacy_full_sort",
            case.legacy_iterations,
            legacy_elapsed,
        );

        let current_elapsed = run_current_case(
            &runtime,
            &backend,
            &logits,
            case.top_k,
            case.current_iterations,
        );
        print_result(
            case,
            "bounded_selection",
            case.current_iterations,
            current_elapsed,
        );
    }
}

fn run_legacy_case(logits: &[f32], top_k: usize, iterations: usize) -> Duration {
    let mut checksum = 0_u64;
    for _ in 0..2 {
        checksum ^= checksum_weights(
            &legacy_full_sort_top_k_f32(black_box(logits), black_box(top_k))
                .expect("legacy warmup succeeds"),
        );
    }

    let started = Instant::now();
    for _ in 0..iterations {
        checksum ^= checksum_weights(
            &legacy_full_sort_top_k_f32(black_box(logits), black_box(top_k))
                .expect("legacy top-k succeeds"),
        );
    }
    let elapsed = started.elapsed();
    black_box(checksum);
    elapsed
}

fn run_current_case(
    runtime: &tokio::runtime::Runtime,
    backend: &CpuNativeMatvecBackend,
    logits: &[f32],
    top_k: usize,
    iterations: usize,
) -> Duration {
    let mut checksum = 0_u64;
    for _ in 0..2 {
        checksum ^= checksum_weights(
            &runtime
                .block_on(backend.softmax_top_k_f32(black_box(logits), black_box(top_k)))
                .expect("current warmup succeeds"),
        );
    }

    let started = Instant::now();
    for _ in 0..iterations {
        checksum ^= checksum_weights(
            &runtime
                .block_on(backend.softmax_top_k_f32(black_box(logits), black_box(top_k)))
                .expect("current top-k succeeds"),
        );
    }
    let elapsed = started.elapsed();
    black_box(checksum);
    elapsed
}

fn print_result(case: &BenchCase, path: &str, iterations: usize, elapsed: Duration) {
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{:<22} {:<18} {:>10} {:>6} {:>8} {:>12.3} {:>12.1}",
        case.name, path, case.vocab, case.top_k, iterations, total_ms, ns_per_iter
    );
}

fn make_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|index| {
            let mixed = (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            (mixed % 16_384) as f32 / 32.0 - 256.0
        })
        .collect()
}

fn legacy_full_sort_top_k_f32(logits: &[f32], top_k: usize) -> Result<Vec<TopKWeight>, MathError> {
    if top_k == 0 || top_k > logits.len() {
        return Err(MathError::InvalidShape(format!(
            "top_k {top_k} must be in 1..={}",
            logits.len()
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidShape(
            "top-k logits must be finite".to_owned(),
        ));
    }
    let mut selected = logits.iter().copied().enumerate().collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    selected.truncate(top_k);
    let max = selected
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = selected
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp_values.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(MathError::InvalidShape(
            "router softmax denominator is invalid".to_owned(),
        ));
    }
    Ok(selected
        .iter()
        .zip(exp_values.iter_mut())
        .map(|((index, _), value)| TopKWeight {
            index: *index,
            weight: *value / sum,
        })
        .collect())
}

fn assert_same_selection(expected: &[TopKWeight], actual: &[TopKWeight]) {
    assert_eq!(
        actual.iter().map(|item| item.index).collect::<Vec<_>>(),
        expected.iter().map(|item| item.index).collect::<Vec<_>>()
    );
    for (expected, actual) in expected.iter().zip(actual) {
        assert_eq!(actual.weight.to_bits(), expected.weight.to_bits());
    }
}

fn checksum_weights(weights: &[TopKWeight]) -> u64 {
    weights.iter().fold(0_u64, |checksum, item| {
        checksum
            ^ (item.index as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
            ^ u64::from(item.weight.to_bits())
    })
}
