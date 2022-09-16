use std::sync::Mutex;
use std::thread;

use alloc::vec::Vec;

use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        CostFunction, Executor, Gradient,
    },
    solver::{
        linesearch::{condition::ArmijoCondition, BacktrackingLineSearch, MoreThuenteLineSearch},
        quasinewton::LBFGS,
    },
};
use hashbrown::HashMap;

use crate::errors::Result;
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
    l2_lambda: Option<f64>,
}

impl<'a> LatticesLoss<'a> {
    fn new(
        lattices: &'a [Lattice],
        provider: &'a FeatureProvider,
        unigram_fids: &'a [u32],
        bigram_fids: &'a [HashMap<u32, u32>],
        n_threads: usize,
        l2_lambda: Option<f64>,
    ) -> Self {
        Self {
            lattices,
            provider,
            unigram_fids,
            bigram_fids,
            n_threads,
            l2_lambda,
        }
    }
}

impl<'a> CostFunction for LatticesLoss<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let (s, r) = crossbeam_channel::unbounded();
        for lattice in self.lattices {
            s.send(lattice).unwrap();
        }
        let mut loss_total = thread::scope(|scope| {
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
        });

        if let Some(lambda) = self.l2_lambda {
            let mut norm2 = 0.0;
            for &p in param {
                norm2 += p * p;
            }
            loss_total += lambda * norm2 * 0.5;
        }

        Ok(loss_total)
    }
}

impl<'a> Gradient for LatticesLoss<'a> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let (s, r) = crossbeam_channel::unbounded();
        for lattice in self.lattices {
            s.send(lattice).unwrap();
        }
        let gradients = Mutex::new(vec![0.0; param.len()]);
        thread::scope(|scope| {
            for _ in 0..self.n_threads {
                scope.spawn(|| {
                    let mut alphas = vec![];
                    let mut betas = vec![];
                    let mut local_gradients = vec![0.0; param.len()];
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
                    #[allow(clippy::significant_drop_in_scrutinee)]
                    for (y, x) in gradients.lock().unwrap().iter_mut().zip(local_gradients) {
                        *y += x;
                    }
                });
            }
        });
        let mut gradients = gradients.into_inner().unwrap();

        if let Some(lambda) = self.l2_lambda {
            for (g, p) in gradients.iter_mut().zip(param) {
                *g += lambda * *p;
            }
        }

        Ok(gradients)
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
    let l2_lambda = (regularization == Regularization::L2).then_some(lambda);
    let loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_fids,
        bigram_fids,
        n_threads,
        l2_lambda,
    );
    match regularization {
        Regularization::L1 => {
            let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4).unwrap())
                .rho(0.5)
                .unwrap();
            let solver = LBFGS::new(linesearch, 7)
                .with_l1_regularization(lambda)
                .unwrap();
            let res = Executor::new(loss_function, solver)
                .configure(|state| state.param(weights_init.to_vec()).max_iters(max_iter))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();
            res.state.param.unwrap()
        }
        Regularization::L2 => {
            let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
            let solver = LBFGS::new(linesearch, 7);
            let res = Executor::new(loss_function, solver)
                .configure(|state| state.param(weights_init.to_vec()).max_iters(max_iter))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();
            res.state.param.unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::{self, hashmap, logsumexp};

    // 0     1     2     3     4     5
    //  /-1-\ /-2-\ /----3----\ /-4-\
    // *     *     *     *     *     *
    //  \----5----/ \-6-/ \-7-/
    // weights:
    // 0->1: 4 (0-1:1 0-2:3)
    // 0->5: 6 (0-2:3 0-2:3)
    // 1->2: 30 (1-4:13 2-3:17)
    // 2->3: 48 (3-2:21 4-3:27)
    // 2->6: 18 (3-4:13 4-1:5)
    // 5->3: 88 (2-2:46 3-3:42)
    // 5->6: 38 (2-4:18 3-1:20)
    // 6->7: 45 (2-3:17 4-4:6)
    // 3->4: 31 (1-2:11 3-1:20)
    // 7->4: 36 (4-2:26 1-1:10)
    // 4->0: 33 (1-0:9 4-0:24)
    // 1: 6
    // 2: 14
    // 3: 8
    // 4: 10
    // 5: 10
    // 6: 10
    // 7: 10
    //
    // 1-2-3-4: 184 *
    // 1-2-6-7-4: 194
    // 5-3-4: 186
    // 5-6-7-4: 176
    //
    // loss = logsumexp(184,194,186,176) - 184
    #[test]
    fn test_loss() {
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 46.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 42.0, 13.0, 24.0, 5.0, 26.0, 27.0, 6.0,
        ];
        let provider = test_utils::generate_test_feature_provider();
        let lattices = vec![test_utils::generate_test_lattice()];
        let unigram_fids = &[1, 3, 5, 7];
        let bigram_fids = &[
            hashmap![0 => 28, 1 => 0, 2 => 2, 3 => 4, 4 => 6],
            hashmap![0 => 8, 1 => 9, 2 => 10, 3 => 11, 4 => 12],
            hashmap![0 => 13, 1 => 14, 2 => 15, 3 => 16, 4 => 17],
            hashmap![0 => 18, 1 => 19, 2 => 20, 3 => 21, 4 => 22],
            hashmap![0 => 23, 1 => 24, 2 => 25, 3 => 26, 4 => 27],
        ];
        let loss_function = LatticesLoss::new(
            &lattices,
            &provider,
            unigram_fids,
            bigram_fids,
            1,
            None,
        );

        let expected = logsumexp!(184.0, 194.0, 186.0, 176.0) - 184.0;
        let result = loss_function.cost(&weights).unwrap();

        assert!((expected - result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gradient() {
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 46.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 42.0, 13.0, 24.0, 5.0, 26.0, 27.0, 6.0,
        ];
        let provider = test_utils::generate_test_feature_provider();
        let lattices = vec![test_utils::generate_test_lattice()];
        let unigram_fids = &[1, 3, 5, 7];
        let bigram_fids = &[
            hashmap![0 => 28, 1 => 0, 2 => 2, 3 => 4, 4 => 6],
            hashmap![0 => 8, 1 => 9, 2 => 10, 3 => 11, 4 => 12],
            hashmap![0 => 13, 1 => 14, 2 => 15, 3 => 16, 4 => 17],
            hashmap![0 => 18, 1 => 19, 2 => 20, 3 => 21, 4 => 22],
            hashmap![0 => 23, 1 => 24, 2 => 25, 3 => 26, 4 => 27],
        ];
        let loss_function = LatticesLoss::new(
            &lattices,
            &provider,
            unigram_fids,
            bigram_fids,
            1,
            None,
        );

        let z = logsumexp!(184.0, 194.0, 186.0, 176.0);
        let prob1 = (184.0 - z).exp();
        let prob2 = (194.0 - z).exp();
        let prob3 = (186.0 - z).exp();
        let prob4 = (176.0 - z).exp();

        let mut expected = vec![0.0; 29];
        // unigram gradients
        for i in [1, 3, 5, 7, 1, 5, 7, 1] {
            expected[i] -= 1.0;
        }
        for i in [1, 3, 5, 7, 1, 5, 7, 1] {
            expected[i] += prob1;
        }
        for i in [1, 3, 5, 7, 1, 7, 3, 5, 7, 1] {
            expected[i] += prob2;
        }
        for i in [3, 5, 1, 5, 7, 1] {
            expected[i] += prob3;
        }
        for i in [3, 5, 1, 7, 3, 5, 7, 1] {
            expected[i] += prob4;
        }
        // bigram gradients
        for i in [0, 2, 12, 16, 20, 26, 10, 19, 8, 23] {
            expected[i] -= 1.0;
        }
        for i in [0, 2, 12, 16, 20, 26, 10, 19, 8, 23] {
            expected[i] += prob1;
        }
        for i in [0, 2, 12, 16, 22, 24, 16, 27, 25, 9, 8, 23] {
            expected[i] += prob2;
        }
        for i in [2, 2, 15, 21, 10, 19, 8, 23] {
            expected[i] += prob3;
        }
        for i in [2, 2, 17, 19, 16, 27, 25, 9, 8, 23] {
            expected[i] += prob4;
        }

        let result = loss_function.gradient(&weights).unwrap();

        let norm = expected
            .iter()
            .zip(&result)
            .fold(0.0, |acc, (a, b)| acc + (a - b).abs());

        assert!(norm < 1e-12);
    }
}
