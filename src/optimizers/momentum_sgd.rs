//! Module for Momentum SGD optimizer.

use core::num::NonZeroU32;

use alloc::vec::Vec;

use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};

use hashbrown::HashMap;

use crate::feature::FeatureProvider;
use crate::lattice::Lattice;
use crate::trainer::{LatticesLoss, Regularization};

pub enum LearningRateDecay {
    Inverse,
    Exponential(f64),
}

#[allow(clippy::too_many_arguments)]
pub fn optimize(
    lattices: &[Lattice],
    provider: &FeatureProvider,
    unigram_weight_indices: &[Option<NonZeroU32>],
    bigram_weight_indices: &[HashMap<u32, u32>],
    weights_init: Vec<f64>,
    _regularization: Regularization,
    lambda: f64,
    max_iter: u64,
    n_threads: usize,

    // Adam parameters
    batch_size: usize,
    momentum: f64,
    eta: f64,
    learning_rate_decay: LearningRateDecay,
) -> Vec<f64> {
    let mut loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
        n_threads,
        None,
    );

    let mut weights = weights_init;
    let mut m = vec![0.0; weights.len()];

    let start_time = std::time::Instant::now();

    let mut cnt = 0;
    for i in 0..max_iter {
        loss_function.shuffle();
        let mut start = 0;
        while start < lattices.len() {
            if cnt % 100 == 0 {
                let cost = loss_function.cost(&weights);
                let elapsed = start_time.elapsed();
                eprintln!("elapsed={}, cost={}", elapsed.as_secs_f32(), cost);
            }

            let grad = loss_function
                .gradient_partial(&weights, start..lattices.len().min(start + batch_size));
            let learning_rate = match learning_rate_decay {
                LearningRateDecay::Inverse => {
                    1.0 / ((i + 1) as f64 + start as f64 / lattices.len() as f64)
                }
                LearningRateDecay::Exponential(alpha) => {
                    alpha.powf(i as f64 + start as f64 / lattices.len() as f64)
                }
            };
            m = m.mul(&momentum).sub(&grad.mul(&(learning_rate * eta)));
            weights = weights.add(&m).sub(&weights.mul(&lambda));
            start += batch_size;
            cnt += 1;
        }
    }
    weights
}
