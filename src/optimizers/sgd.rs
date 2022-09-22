use core::num::NonZeroU32;

use alloc::vec::Vec;

use argmin::{
    core::{CostFunction, Executor, Gradient, LineSearch},
    solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch},
};
use argmin_math::{ArgminL1Norm, ArgminMul};
use hashbrown::HashMap;

use crate::feature::FeatureProvider;
use crate::lattice::Lattice;
use crate::trainer::{LatticesLoss, Regularization};

#[allow(clippy::too_many_arguments)]
pub fn optimize(
    lattices: &[Lattice],
    provider: &FeatureProvider,
    unigram_weight_indices: &[Option<NonZeroU32>],
    bigram_weight_indices: &[HashMap<u32, u32>],
    weights_init: Vec<f64>,
    regularization: Regularization,
    lambda: f64,
    batch_size: usize,
    max_iter: u64,
    n_threads: usize,
) -> Vec<f64> {
    let mut loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
        n_threads,
        (regularization == Regularization::L2).then_some(lambda),
    );
    // Optimizes initial learning rate
    let cost = loss_function.cost(&weights_init).unwrap();
    let grad = loss_function.gradient(&weights_init).unwrap();
    let mut linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4).unwrap())
        .rho(0.5)
        .unwrap();
    linesearch.search_direction(grad.mul(&-1.0));
    let result = Executor::new(loss_function.clone(), linesearch)
        .configure(|config| {
            config
                .param(weights_init.clone())
                .gradient(grad.clone())
                .cost(cost)
        })
        .ctrlc(false)
        .run()
        .unwrap();

    let init_learning_rate = result.state.param.unwrap().l1_norm() / grad.l1_norm();

    let mut weights = weights_init;
    for i in 0..max_iter {
        loss_function.shuffle();
        let mut start = 0;
        while start < lattices.len() {
            let grad = loss_function
                .gradient_partial(&weights, start..lattices.len().min(start + batch_size));
            let learning_rate =
                init_learning_rate / ((i + 1) as f64 + start as f64 / lattices.len() as f64);
            for (w, g) in weights.iter_mut().zip(&grad) {
                *w -= g * learning_rate;
            }
            start += batch_size;
        }
    }
    weights
}
