//! Module for SGD optimizer.

use core::num::NonZeroU32;

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::feature::FeatureProvider;
use crate::lattice::Lattice;
use crate::trainer::{LatticesLoss, Regularization};
use crate::optimizers::LearningRateDecay;

#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize(
    lattices: &[Lattice],
    provider: &FeatureProvider,
    unigram_weight_indices: &[Option<NonZeroU32>],
    bigram_weight_indices: &[HashMap<u32, u32>],
    weights_init: Vec<f64>,
    regularization: Regularization,
    lambda: f64,
    max_iter: u64,
    stop_criteria: f64,
    n_threads: usize,

    // SGD parameters
    batch_size: usize,
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

    let start_time = std::time::Instant::now();

    let mut prev_cost = f64::INFINITY;
    for epoch in 0..max_iter {
        let cost = loss_function.cost(&weights);
        let cost = cost + lambda * match regularization {
            Regularization::L1 => {
                let mut reg = 0.0;
                for w in &weights {
                    reg += w.abs();
                }
                reg
            }
            Regularization::L2 => {
                let mut reg = 0.0;
                for &w in &weights {
                    reg += w * w;
                }
                reg / 2.0
            }
        };
        let elapsed = start_time.elapsed().as_secs_f32();
        eprintln!("epoch={epoch}, elapsed={elapsed}, cost={cost}");
        if (cost - prev_cost).abs() / prev_cost < stop_criteria.abs() {
            break;
        }
        prev_cost = cost;

        loss_function.shuffle();
        let mut start = 0;
        while start < lattices.len() {
            let learning_rate = eta * match learning_rate_decay {
                LearningRateDecay::Inverse => {
                    1.0 / ((epoch + 1) as f64 + start as f64 / lattices.len() as f64)
                }
                LearningRateDecay::Exponential(alpha) => {
                    alpha.powf(epoch as f64 + start as f64 / lattices.len() as f64)
                }
            };

            let range = start..lattices.len().min(start + batch_size);
            let grad = loss_function.gradient_partial(&weights, range.clone());
            for (w, g) in weights.iter_mut().zip(&grad) {
                *w -= learning_rate * g;
            }
            let reg_factor = learning_rate * lambda * range.len() as f64 / lattices.len() as f64;
            match regularization {
                Regularization::L1 => {
                    for w in &mut weights {
                        if w.is_sign_positive() {
                            *w = (*w - reg_factor).max(0.0);
                        } else {
                            *w = (*w + reg_factor).min(0.0);
                        }
                    }
                }
                Regularization::L2 => {
                    for w in &mut weights {
                        *w -= reg_factor * *w;
                    }
                }
            }
            start += batch_size;
        }
    }
    weights
}
