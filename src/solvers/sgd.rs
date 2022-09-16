use std::sync::Mutex;
use std::thread;

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::vector::{RegularizedWeightVector, SparseGrdientVector, WeightVector};
use crate::feature::FeatureProvider;
use crate::forward_backward;
use crate::lattice::Lattice;
use crate::trainer::Regularization;

struct LatticesLoss<'a> {
    lattices: &'a [Lattice],
    provider: &'a FeatureProvider,
    unigram_fids: &'a [u32],
    bigram_fids: &'a [HashMap<u32, u32>],
    n_threads: usize,
}

impl<'a> LatticesLoss<'a> {
    fn new(
        lattices: &'a [Lattice],
        provider: &'a FeatureProvider,
        unigram_fids: &'a [u32],
        bigram_fids: &'a [HashMap<u32, u32>],
        n_threads: usize,
    ) -> Self {
        Self {
            lattices,
            provider,
            unigram_fids,
            bigram_fids,
            n_threads,
        }
    }

    fn cost<F>(&self, param: &RegularizedWeightVector<F>) -> f64 where F: Fn(f64, usize, usize) -> f64 + Send + Sync {
        let (s, r) = crossbeam_channel::unbounded();
        for lattice in self.lattices {
            s.send(lattice).unwrap();
        }
        thread::scope(|scope| {
            let mut threads = vec![];
            for _ in 0..self.n_threads {
                let t = scope.spawn(|| {
                    let mut alphas = vec![];
                    let mut betas = vec![];
                    let mut loss_total = 0.0;
                    while let Ok(lattice) = r.try_recv() {
                        let z = forward_backward::calculate_alphas_betas(
                            lattice,
                            self.provider,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            &mut alphas,
                            &mut betas,
                        );
                        let loss = forward_backward::calculate_loss(
                            lattice,
                            self.provider,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            z,
                        );
                        loss_total += loss;
                    }
                    loss_total
                });
                threads.push(t);
            }
            let mut loss_total = 0.0;
            for t in threads {
                let loss = t.join().unwrap();
                loss_total += loss;
            }
            loss_total
        })
    }

    fn gradient<F>(&self, param: &RegularizedWeightVector<F>, select: &[usize]) -> SparseGrdientVector where F: Fn(f64, usize, usize) -> f64 + Send + Sync{
        let (s, r) = crossbeam_channel::unbounded();
        for &id in select {
            s.send(&self.lattices[id]).unwrap();
        }
        let gradients = Mutex::new(SparseGrdientVector::new());
        thread::scope(|scope| {
            for _ in 0..self.n_threads {
                scope.spawn(|| {
                    let mut alphas = vec![];
                    let mut betas = vec![];
                    let mut local_gradients = SparseGrdientVector::new();
                    while let Ok(lattice) = r.try_recv() {
                        let z = forward_backward::calculate_alphas_betas(
                            lattice,
                            self.provider,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            &mut alphas,
                            &mut betas,
                        );
                        forward_backward::update_gradient(
                            lattice,
                            self.provider,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            &alphas,
                            &betas,
                            z,
                            &mut local_gradients,
                        );
                    }
                    local_gradients.merge_gradients(&mut *gradients.lock().unwrap());
                });
            }
        });
        gradients.into_inner().unwrap()
    }
}

pub fn solve(
    lattices: &[Lattice],
    provider: &FeatureProvider,
    unigram_fids: &[u32],
    bigram_fids: &[HashMap<u32, u32>],
    weights_init: &[f64],
    regularization: Regularization,
    lambda: f64,
    max_iter: u64,
    n_threads: usize,
) -> Vec<f64> {
    let loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_fids,
        bigram_fids,
        n_threads,
    );

    let eta = 1e-2;

    match regularization {
        Regularization::L1 => {
            let mut param = RegularizedWeightVector::new(weights_init.len(), |w, t, last_update| {
                if w.is_sign_positive() {
                    (w - (t - last_update) as f64 * lambda * eta).max(0.0)
                } else {
                    (w + (t - last_update) as f64 * lambda * eta).min(0.0)
                }
            });
            for _ in 0..max_iter {
                for i in 0..lattices.len() {
                    let mut grad = loss_function.gradient(&param, &[i]);
                    grad.apply_gradients(&mut param, eta);
                    param.increment_step();
                }
                let reg: f64 = (0..weights_init.len()).map(|i| param.get_weight(i).abs()).sum();
                dbg!(loss_function.cost(&param) + lambda * reg);
            }
            param.into()
        }
        Regularization::L2 => {
            let mut param = RegularizedWeightVector::new(weights_init.len(), |w, t, last_update| {
                w * (1.0 - lambda * eta).powf((t - last_update) as f64)
            });
            for _ in 0..max_iter {
                for i in 0..lattices.len() {
                    let mut grad = loss_function.gradient(&param, &[i]);
                    grad.apply_gradients(&mut param, eta);
                    param.increment_step();
                }
                let reg: f64 = (0..weights_init.len()).map(|i| param.get_weight(i).abs().powi(2)).sum();
                dbg!(loss_function.cost(&param) + lambda * reg);
            }
            param.into()
        }
    }
}
