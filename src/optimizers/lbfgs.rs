use core::num::NonZeroU32;

use alloc::vec::Vec;

use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        Executor,
    },
    solver::{
        linesearch::{condition::ArmijoCondition, BacktrackingLineSearch, MoreThuenteLineSearch},
        quasinewton::LBFGS,
    },
};
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
    max_iter: u64,
    n_threads: usize,
) -> Vec<f64> {
    let loss_function = LatticesLoss::new(
        lattices,
        provider,
        unigram_weight_indices,
        bigram_weight_indices,
        n_threads,
        (regularization == Regularization::L2).then_some(lambda),
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
                .configure(|state| state.param(weights_init).max_iters(max_iter))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();
            res.state.param.unwrap()
        }
        Regularization::L2 => {
            let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
            let solver = LBFGS::new(linesearch, 7);
            let res = Executor::new(loss_function, solver)
                .configure(|state| state.param(weights_init).max_iters(max_iter))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();
            res.state.param.unwrap()
        }
    }
}
