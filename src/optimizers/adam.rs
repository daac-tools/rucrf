//! Module for Adam optimizer.

use core::num::NonZeroU32;

use alloc::vec::Vec;

use argmin_math::{ArgminAdd, ArgminDiv, ArgminMul};
use hashbrown::HashMap;

use crate::feature::FeatureProvider;
use crate::lattice::Lattice;
use crate::trainer::LatticesLoss;

#[allow(clippy::too_many_arguments)]
pub fn optimize(
    lattices: &[Lattice],
    provider: &FeatureProvider,
    unigram_weight_indices: &[Option<NonZeroU32>],
    bigram_weight_indices: &[HashMap<u32, u32>],
    weights_init: &[f64],
    max_iter: u64,
    n_threads: usize,

    // Adam parameters
    batch_size: usize,
    eta0: f64,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) -> Vec<f64> {
    let mut weights = weights_init.to_vec();
    let mut loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
        n_threads,
        None,
    );

    let mut m = vec![0.0; weights.len()];
    let mut v = vec![0.0; weights.len()];
    let mut step = 1;
    for i in 0..max_iter {
        loss_function.shuffle();
        let mut st = 0;
        while st < lattices.len() {
            let eta = eta0.powf(i as f64 + st as f64 / lattices.len() as f64);
            let g =
                loss_function.gradient_partial(&weights, st..(st + batch_size).min(lattices.len()));
            let g2 = g.mul(&g);
            m = m.mul(&beta1).add(&g.mul(&(1.0 - beta1)));
            v = v.mul(&beta2).add(&g2.mul(&(1.0 - beta2)));
            let hat_m = m.div(&(1.0 - beta1.powi(step)));
            let hat_v = v.div(&(1.0 - beta2.powi(step)));
            let diff = hat_m
                .div(
                    &hat_v
                        .add(&eps)
                        .into_iter()
                        .map(f64::sqrt)
                        .collect::<Vec<_>>(),
                )
                .mul(&alpha);
            for (w, d) in weights.iter_mut().zip(&diff) {
                *w -= eta * d;
            }
            if step % 100 == 0 {
                dbg!(loss_function.cost(&weights));
            }
            st += batch_size;
            step += 1;
        }
    }

    weights
}
