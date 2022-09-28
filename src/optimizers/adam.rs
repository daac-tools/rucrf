//! Module for Adam optimizer.

use core::num::NonZeroU32;

use alloc::vec::Vec;

use argmin_math::{ArgminMul, ArgminAdd, ArgminDiv};
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
    weights_init: &[f64],
    regularization: Regularization,
    lambda: f64,
    max_iter: u64,
    n_threads: usize,

    // Adam parameters
    batch_size: usize,
    eta: f64,
    rho1: f64,
    rho2: f64,
    eps: f64,
) -> Vec<f64> {
    let mut weights = weights_init.to_vec();
    let mut loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
        n_threads,
        (regularization == Regularization::L2).then_some(lambda),
    );

    let mut m = vec![];
    let mut v = vec![];
    let mut step = 1;
    for _ in 0..max_iter {
        loss_function.shuffle();
        let mut st = 0;
        while st < lattices.len() {
            let g = loss_function.gradient_partial(&weights, st..(st+batch_size).min(lattices.len()));
            let g2 = g.mul(&g);
            if m.is_empty() {
                m = g;
            } else {
                m = m.mul(&rho1).add(&g.mul(&(1.0 - rho1)));
            }
            if v.is_empty() {
                v = g2;
            } else {
                v = v.mul(&rho2).add(&g2.mul(&(1.0 - rho2)));
            }
            let hat_m = m.div(&(1.0 - rho1.powi(step)));
            let hat_v = m.div(&(1.0 - rho2.powi(step)));
            let diff = hat_m.div(&hat_v.add(&eps).into_iter().map(f64::sqrt).collect::<Vec<_>>()).mul(&-eta);
            for (w, d) in weights.iter_mut().zip(&diff) {
                *w += d;
            }
            st += batch_size;
            step += 1;
            if step % 10 == 0 {
                dbg!(loss_function.cost(&weights));
            }
        }
    }

    weights
}
