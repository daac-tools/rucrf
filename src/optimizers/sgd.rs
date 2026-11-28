//! Module for the stochastic gradient descent (SGD) optimizer.
//!
//! # References
//!
//! * Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou. 2009.
//!   Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative
//!   Penalty.
//!   In Proceedings of ACL-IJCNLP 2009, pages 477–485.

use std::num::NonZeroU32;
use std::thread;
use std::time::Instant;

use hashbrown::HashMap;

use crate::feature::FeatureProvider;
use crate::forward_backward::{self, Alpha, Beta, Gradients, ScaledWeights, Weights};
use crate::lattice::Lattice;
use crate::trainer::Regularization;

/// Hyperparameters of SGD.
#[derive(Clone, Copy, Debug)]
pub struct SgdConfig {
    /// Initial learning rate.
    pub learning_rate: f64,
    /// The learning rate is multiplied by this value per epoch.
    pub learning_rate_decay: f64,
    /// Number of examples in a mini-batch.
    pub batch_size: usize,
    /// Seed of the random number generator used for shuffling examples.
    pub seed: u64,
}

/// A simple pseudo random number generator (SplitMix64).
struct SplitMix64(u64);

impl SplitMix64 {
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a random number in `0..n`.
    fn gen_index(&mut self, n: usize) -> usize {
        let r = u128::from(self.next_u64()) * u128::try_from(n).unwrap();
        usize::try_from(r >> 64).unwrap()
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.gen_index(i + 1);
            items.swap(i, j);
        }
    }
}

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
#[inline(always)]
const fn usize_to_f64(x: usize) -> f64 {
    x as f64
}

/// Sparse gradient vector backed by a dense buffer.
struct SparseGradient {
    values: Vec<f64>,
    touched: Vec<bool>,
    indices: Vec<usize>,
}

impl SparseGradient {
    fn new(n: usize) -> Self {
        Self {
            values: vec![0.0; n],
            touched: vec![false; n],
            indices: vec![],
        }
    }

    fn clear(&mut self) {
        for &i in &self.indices {
            self.values[i] = 0.0;
            self.touched[i] = false;
        }
        self.indices.clear();
    }

    /// Adds all values of `other` to `self` and clears `other`.
    fn merge_from(&mut self, other: &mut Self) {
        for &i in &other.indices {
            self.add(i, other.values[i]);
        }
        other.clear();
    }
}

impl Gradients for SparseGradient {
    #[inline(always)]
    fn add(&mut self, i: usize, v: f64) {
        if !self.touched[i] {
            self.touched[i] = true;
            self.indices.push(i);
        }
        self.values[i] += v;
    }
}

/// Per-thread working memory.
struct Worker {
    gradient: SparseGradient,
    alphas: Vec<Vec<Alpha>>,
    betas: Vec<Vec<Beta>>,
}

struct Context<'a> {
    lattices: &'a [Lattice],
    provider: &'a FeatureProvider,
    unigram_weight_indices: &'a [Option<NonZeroU32>],
    bigram_weight_indices: &'a [HashMap<u32, u32>],
}

impl Worker {
    fn new(n: usize) -> Self {
        Self {
            gradient: SparseGradient::new(n),
            alphas: vec![],
            betas: vec![],
        }
    }

    /// Accumulates gradients of the given examples and returns the sum of their losses.
    fn process<W: Weights + ?Sized>(&mut self, ctx: &Context, weights: &W, ids: &[usize]) -> f64 {
        let mut loss_total = 0.0;
        for &id in ids {
            let lattice = &ctx.lattices[id];
            let z = forward_backward::calculate_alphas_betas(
                lattice,
                ctx.provider,
                weights,
                ctx.unigram_weight_indices,
                ctx.bigram_weight_indices,
                &mut self.alphas,
                &mut self.betas,
            );
            loss_total += forward_backward::calculate_loss(
                lattice,
                ctx.provider,
                weights,
                ctx.unigram_weight_indices,
                ctx.bigram_weight_indices,
                z,
            );
            forward_backward::update_gradient(
                lattice,
                ctx.provider,
                weights,
                ctx.unigram_weight_indices,
                ctx.bigram_weight_indices,
                &self.alphas,
                &self.betas,
                z,
                &mut self.gradient,
            );
        }
        loss_total
    }
}

/// Computes the gradient of a mini-batch into `workers[0].gradient` and returns the sum of losses.
fn compute_batch_gradient<W: Weights + Sync + ?Sized>(
    ctx: &Context,
    weights: &W,
    batch: &[usize],
    workers: &mut [Worker],
) -> f64 {
    // Each worker processes a contiguous chunk, and the results are merged in a fixed order,
    // so the result is deterministic for a given number of threads.
    let chunk_size = batch.len().div_ceil(workers.len());
    let loss_total = if workers.len() == 1 || batch.len() == 1 {
        workers[0].process(ctx, weights, batch)
    } else {
        thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .zip(batch.chunks(chunk_size))
                .map(|(worker, ids)| scope.spawn(move || worker.process(ctx, weights, ids)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        })
    };
    let (first, rest) = workers.split_first_mut().unwrap();
    for worker in rest {
        first.gradient.merge_from(&mut worker.gradient);
    }
    loss_total
}

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
    config: SgdConfig,
) -> Vec<f64> {
    let n_weights = weights_init.len();
    let n_examples = lattices.len();
    if n_examples == 0 {
        return weights_init.to_vec();
    }
    let ctx = Context {
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
    };

    let mut rng = SplitMix64(config.seed);
    let mut order: Vec<usize> = (0..n_examples).collect();
    let batch_size = config.batch_size.min(n_examples);
    let n_workers = n_threads.min(batch_size).max(1);
    let mut workers: Vec<_> = (0..n_workers).map(|_| Worker::new(n_weights)).collect();

    // For L1: weights. For L2: values (weights = scale * values).
    let mut weights = weights_init.to_vec();
    let mut scale = 1.0;
    // For L1: the total L1-penalty that each weight could have received (u) and the penalty
    // actually received by each weight (q).
    let mut u = 0.0;
    let mut q = if regularization == Regularization::L1 {
        vec![0.0; n_weights]
    } else {
        vec![]
    };

    let n = usize_to_f64(n_examples);
    let mut n_processed = 0usize;
    for epoch in 0..max_iter {
        let start = Instant::now();
        rng.shuffle(&mut order);
        let mut loss_total = 0.0;
        for batch in order.chunks(batch_size) {
            let eta = config.learning_rate
                * config
                    .learning_rate_decay
                    .powf(usize_to_f64(n_processed) / n);
            let step = eta / usize_to_f64(batch.len());
            n_processed += batch.len();
            match regularization {
                Regularization::L1 => {
                    loss_total +=
                        compute_batch_gradient(&ctx, weights.as_slice(), batch, &mut workers);
                    u += eta * lambda / n;
                    let gradient = &workers[0].gradient;
                    for &i in &gradient.indices {
                        let w = step.mul_add(-gradient.values[i], weights[i]);
                        let new_w = if w > 0.0 {
                            (w - (u + q[i])).max(0.0)
                        } else if w < 0.0 {
                            (w + (u - q[i])).min(0.0)
                        } else {
                            w
                        };
                        q[i] += new_w - w;
                        weights[i] = new_w;
                    }
                }
                Regularization::L2 => {
                    let scaled = ScaledWeights {
                        values: &weights,
                        scale,
                    };
                    loss_total += compute_batch_gradient(&ctx, &scaled, batch, &mut workers);
                    // Clamps the decay factor to avoid zero or negative scale.
                    scale *= (1.0 - eta * lambda / n).max(1e-3);
                    let gradient = &workers[0].gradient;
                    let step = step / scale;
                    for &i in &gradient.indices {
                        weights[i] = step.mul_add(-gradient.values[i], weights[i]);
                    }
                    if scale < 1e-9 {
                        for w in &mut weights {
                            *w *= scale;
                        }
                        scale = 1.0;
                    }
                }
            }
            workers[0].gradient.clear();
        }

        let (regularizer, n_nonzero) = match regularization {
            Regularization::L1 => (
                lambda * weights.iter().map(|w| w.abs()).sum::<f64>(),
                weights.iter().filter(|&&w| w != 0.0).count(),
            ),
            Regularization::L2 => (
                lambda * 0.5 * scale * scale * weights.iter().map(|w| w * w).sum::<f64>(),
                weights.iter().filter(|&&w| w != 0.0).count(),
            ),
        };
        // `running_loss` is the sum of losses computed during the epoch with the weights being
        // updated, so it is not the exact loss at the end of the epoch.
        eprintln!(
            "epoch: {}, running_loss: {}, regularizer: {}, eta: {}, nonzero: {}/{}, time: {:.3}",
            epoch,
            loss_total,
            regularizer,
            config.learning_rate
                * config
                    .learning_rate_decay
                    .powf(usize_to_f64(n_processed) / n),
            n_nonzero,
            n_weights,
            start.elapsed().as_secs_f64(),
        );
    }

    if regularization == Regularization::L2 {
        for w in &mut weights {
            *w *= scale;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_is_permutation() {
        let mut rng = SplitMix64(42);
        let mut items: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut items);
        assert_ne!((0..100).collect::<Vec<_>>(), items);
        items.sort_unstable();
        assert_eq!((0..100).collect::<Vec<_>>(), items);
    }

    #[test]
    fn test_sparse_gradient_merge() {
        let mut a = SparseGradient::new(5);
        let mut b = SparseGradient::new(5);
        a.add(1, 1.0);
        a.add(3, 2.0);
        b.add(3, 0.5);
        b.add(4, -1.0);
        a.merge_from(&mut b);
        assert_eq!(vec![0.0, 1.0, 0.0, 2.5, -1.0], a.values);
        let mut indices = a.indices.clone();
        indices.sort_unstable();
        assert_eq!(vec![1, 3, 4], indices);
        assert!(b.indices.is_empty());
        assert_eq!(vec![0.0; 5], b.values);
        assert_eq!(vec![false; 5], b.touched);
    }
}
