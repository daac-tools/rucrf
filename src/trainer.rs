use std::sync::Mutex;
use std::thread;

use alloc::vec::Vec;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;

use crate::forward_backward;
use crate::lattice::Lattice;
use crate::model::Model;
use crate::Result;

struct LatticesLoss<'a> {
    lattices: &'a [Lattice],
    unigram_fids: Vec<HashMap<usize, usize>>,
    bigram_fids: Vec<HashMap<usize, usize>>,
    n_threads: usize,
    l2_lambda: Option<f64>,
}

impl<'a> LatticesLoss<'a> {
    fn new(
        lattices: &'a [Lattice],
        unigram_fids: Vec<HashMap<usize, usize>>,
        bigram_fids: Vec<HashMap<usize, usize>>,
        n_threads: usize,
        l2_lambda: Option<f64>,
    ) -> Self {
        Self {
            lattices,
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
        let (mut loss_total, p) = thread::scope(|scope| {
            let mut threads = vec![];
            for _ in 0..self.n_threads {
                let t = scope.spawn(|| {
                    let mut alphas = vec![];
                    let mut betas = vec![];
                    let mut loss_total = 0.0;
                    let mut p = 0.0;
                    while let Ok(lattice) = r.try_recv() {
                        let z = forward_backward::calculate_alphas_betas(
                            lattice,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            &mut alphas,
                            &mut betas,
                        );
                        let loss = forward_backward::calculate_loss(
                            lattice,
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            z,
                        );
                        p += (-loss).exp();
                        loss_total += loss;
                    }
                    (loss_total, p)
                });
                threads.push(t);
            }
            let mut loss_total = 0.0;
            let mut p_total = 0.0;
            for t in threads {
                let (loss, p) = t.join().unwrap();
                loss_total += loss;
                p_total += p;
            }
            (loss_total, p_total / self.lattices.len() as f64)
        });

        eprintln!("loss = {loss_total}");
        eprintln!("mean likelihood = {p}");

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
                            param,
                            &self.unigram_fids,
                            &self.bigram_fids,
                            &mut alphas,
                            &mut betas,
                        );
                        forward_backward::update_gradient(
                            lattice,
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

/// L1- or L2- regularization settings
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub enum Regularization {
    /// Performs L1-regularization.
    L1,

    /// Performs L2-regularization.
    L2,
}

/// Trainer for CRF
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Trainer {
    max_iter: u64,
    n_threads: usize,
    regularization: Regularization,
    lambda: f64,
}

impl Trainer {
    /// Creates a new trainer
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            n_threads: 1,
            regularization: Regularization::L2,
            lambda: 0.1,
        }
    }

    /// Sets the maximum number of iterations
    pub fn max_iter(mut self, max_iter: u64) -> Result<Self> {
        if max_iter == 0 {
            return Err("max_iter must not be 0");
        }
        self.max_iter = max_iter;
        Ok(self)
    }

    /// Sets regularization settings.
    pub fn regularization(mut self, regularization: Regularization, lambda: f64) -> Result<Self> {
        if lambda < 0.0 {
            return Err("lambda must be greater than or equal to 0.0");
        }
        self.regularization = regularization;
        self.lambda = lambda;
        Ok(self)
    }

    /// Sets the number of threads
    pub fn n_threads(mut self, n_threads: usize) -> Result<Self> {
        if n_threads == 0 {
            return Err("n_threads must not be 0");
        }
        self.n_threads = n_threads;
        Ok(self)
    }

    #[inline(always)]
    fn update_unigram_feature(
        label: usize,
        feature_id: usize,
        unigram_fids: &mut Vec<HashMap<usize, usize>>,
        weights: &mut Vec<f64>,
    ) {
        if unigram_fids.len() <= label {
            unigram_fids.resize(label + 1, HashMap::new());
        }
        let features = &mut unigram_fids[label];
        if let RawEntryMut::Vacant(v) = features.raw_entry_mut().from_key(&feature_id) {
            v.insert(feature_id, weights.len());
            weights.push(0.0);
        }
    }

    #[inline(always)]
    fn update_bigram_feature(
        left_label: usize,
        right_label: usize,
        bigram_fids: &mut Vec<HashMap<usize, usize>>,
        weights: &mut Vec<f64>,
    ) {
        if bigram_fids.len() <= left_label {
            bigram_fids.resize(left_label + 1, HashMap::new());
        }
        let features = &mut bigram_fids[left_label];
        if let RawEntryMut::Vacant(v) = features.raw_entry_mut().from_key(&right_label) {
            v.insert(right_label, weights.len());
            weights.push(0.0);
        }
    }

    fn update_features(
        lattice: &Lattice,
        unigram_fids: &mut Vec<HashMap<usize, usize>>,
        bigram_fids: &mut Vec<HashMap<usize, usize>>,
        weights: &mut Vec<f64>,
    ) {
        for (i, node) in lattice.nodes().iter().enumerate() {
            if i == 0 {
                for curr_edge in node.edges() {
                    Self::update_bigram_feature(0, curr_edge.label, bigram_fids, weights);
                }
            }
            for curr_edge in node.edges() {
                for next_edge in lattice.nodes()[curr_edge.target()].edges() {
                    Self::update_bigram_feature(
                        curr_edge.label,
                        next_edge.label,
                        bigram_fids,
                        weights,
                    );
                }
                if curr_edge.target() == lattice.nodes().len() - 1 {
                    Self::update_bigram_feature(curr_edge.label, 0, bigram_fids, weights);
                }
                if let Some(feature_ids) = lattice.features().get(&(i, curr_edge.target())) {
                    for feature_id in feature_ids {
                        Self::update_unigram_feature(
                            curr_edge.label,
                            feature_id.feature_id,
                            unigram_fids,
                            weights,
                        );
                    }
                }
            }
        }
    }

    /// Starts training and generates a model from the given lattices
    pub fn train(&self, lattices: &[Lattice]) -> Model {
        let mut unigram_fids = vec![];
        let mut bigram_fids = vec![];
        let mut weights_init = vec![];

        for lattice in lattices {
            Self::update_features(
                lattice,
                &mut unigram_fids,
                &mut bigram_fids,
                &mut weights_init,
            );
        }

        let weights;
        let loss_function_used;
        match self.regularization {
            Regularization::L1 => {
                let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4).unwrap())
                    .rho(0.5)
                    .unwrap();
                let loss_function =
                    LatticesLoss::new(lattices, unigram_fids, bigram_fids, self.n_threads, None);
                let solver = LBFGS::new(linesearch, 7)
                    .with_l1_regularization(self.lambda)
                    .unwrap();
                let res = Executor::new(loss_function, solver)
                    .configure(|state| state.param(weights_init).max_iters(self.max_iter))
                    .add_observer(SlogLogger::term(), ObserverMode::Always)
                    .run()
                    .unwrap();
                weights = res.state.param.unwrap();
                loss_function_used = res.problem.problem.unwrap();
            }
            Regularization::L2 => {
                let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
                let loss_function = LatticesLoss::new(
                    lattices,
                    unigram_fids,
                    bigram_fids,
                    self.n_threads,
                    Some(self.lambda),
                );
                let solver = LBFGS::new(linesearch, 7);
                let res = Executor::new(loss_function, solver)
                    .configure(|state| state.param(weights_init).max_iters(self.max_iter))
                    .add_observer(SlogLogger::term(), ObserverMode::Always)
                    .run()
                    .unwrap();
                weights = res.state.param.unwrap();
                loss_function_used = res.problem.problem.unwrap();
            }
        }

        let mut unigram_fids = vec![];
        let mut bigram_fids = vec![];
        let mut new_weights = vec![];
        for hm in loss_function_used.unigram_fids {
            let mut new_hm = HashMap::new();
            for (k, v) in hm {
                let w = weights[v];
                if w.abs() > f64::EPSILON {
                    new_hm.insert(k, new_weights.len());
                    new_weights.push(w);
                }
            }
            unigram_fids.push(new_hm);
        }
        for hm in loss_function_used.bigram_fids {
            let mut new_hm = HashMap::new();
            for (k, v) in hm {
                let w = weights[v];
                if w.abs() > f64::EPSILON {
                    new_hm.insert(k, new_weights.len());
                    new_weights.push(w);
                }
            }
            bigram_fids.push(new_hm);
        }

        dbg!(new_weights.len());
        dbg!(weights.len());

        bigram_fids[0].insert(0, new_weights.len());
        new_weights.push(f64::NEG_INFINITY);

        Model {
            weights: new_weights,
            unigram_fids,
            bigram_fids,
        }
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}
