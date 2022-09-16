use core::num::NonZeroU32;

use alloc::vec::Vec;

use hashbrown::{hash_map::RawEntryMut, HashMap, HashSet};

use crate::lattice::Lattice;
use crate::model::RawModel;
use crate::utils::FromU32;
use crate::solvers::{lbfgs, sgd};
use crate::errors::Result;
use crate::errors::RucrfError;
use crate::feature::FeatureProvider;

/// L1- or L2- regularization settings
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Regularization {
    /// Performs L1-regularization.
    L1,

    /// Performs L2-regularization.
    L2,
}

/// CRF trainer.
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Trainer {
    max_iter: u64,
    n_threads: usize,
    regularization: Regularization,
    lambda: f64,
}

impl Trainer {
    /// Creates a new trainer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_iter: 100,
            n_threads: 1,
            regularization: Regularization::L1,
            lambda: 0.1,
        }
    }

    /// Sets the maximum number of iterations.
    ///
    /// # Errors
    ///
    /// `max_iter` must be >= 1.
    pub const fn max_iter(mut self, max_iter: u64) -> Result<Self> {
        if max_iter == 0 {
            return Err(RucrfError::invalid_argument("max_iter must be >= 1"));
        }
        self.max_iter = max_iter;
        Ok(self)
    }

    /// Sets regularization settings.
    ///
    /// # Errors
    ///
    /// `lambda` must be >= 0.
    pub fn regularization(mut self, regularization: Regularization, lambda: f64) -> Result<Self> {
        if lambda < 0.0 {
            return Err(RucrfError::invalid_argument("lambda must be >= 0"));
        }
        self.regularization = regularization;
        self.lambda = lambda;
        Ok(self)
    }

    /// Sets the number of threads.
    ///
    /// # Errors
    ///
    /// `n_threads` must be >= 1.
    pub const fn n_threads(mut self, n_threads: usize) -> Result<Self> {
        if n_threads == 0 {
            return Err(RucrfError::invalid_argument("n_thread must be >= 1"));
        }
        self.n_threads = n_threads;
        Ok(self)
    }

    #[inline(always)]
    fn update_unigram_feature(
        provider: &FeatureProvider,
        label: NonZeroU32,
        unigram_fids: &mut Vec<u32>,
        weights: &mut Vec<f64>,
    ) {
        for &fid in provider.get_feature_set(label).unigram() {
            let fid = usize::from_u32(fid.get());
            if unigram_fids.len() <= fid {
                unigram_fids.resize(fid + 1, 0);
            }
            if unigram_fids.len() == fid + 1 {
                unigram_fids[fid] = u32::try_from(weights.len()).unwrap();
                weights.push(0.0);
            }
        }
    }

    #[inline(always)]
    fn update_bigram_feature(
        provider: &FeatureProvider,
        left_label: Option<NonZeroU32>,
        right_label: Option<NonZeroU32>,
        bigram_fids: &mut Vec<HashMap<u32, u32>>,
        weights: &mut Vec<f64>,
    ) {
        match (left_label, right_label) {
            (Some(left_label), Some(right_label)) => {
                let left_features = provider.get_feature_set(left_label).bigram_left();
                let right_features = provider.get_feature_set(right_label).bigram_right();
                for (left_fid, right_fid) in left_features.iter().zip(right_features) {
                    if let (Some(left_fid), Some(right_fid)) = (left_fid, right_fid) {
                        let left_fid = usize::try_from(left_fid.get()).unwrap();
                        let right_fid = right_fid.get();
                        if bigram_fids.len() <= left_fid {
                            bigram_fids.resize(left_fid + 1, HashMap::new());
                        }
                        let features = &mut bigram_fids[left_fid];
                        if let RawEntryMut::Vacant(v) =
                            features.raw_entry_mut().from_key(&right_fid)
                        {
                            v.insert(right_fid, u32::try_from(weights.len()).unwrap());
                            weights.push(0.0);
                        }
                    }
                }
            }
            (Some(left_label), None) => {
                for left_fid in provider
                    .get_feature_set(left_label)
                    .bigram_left()
                    .iter()
                    .flatten()
                {
                    let left_fid = usize::try_from(left_fid.get()).unwrap();
                    if bigram_fids.len() <= left_fid {
                        bigram_fids.resize(left_fid + 1, HashMap::new());
                    }
                    let features = &mut bigram_fids[left_fid];
                    if let RawEntryMut::Vacant(v) = features.raw_entry_mut().from_key(&0) {
                        v.insert(0, u32::try_from(weights.len()).unwrap());
                        weights.push(0.0);
                    }
                }
            }
            (None, Some(right_label)) => {
                for right_fid in provider
                    .get_feature_set(right_label)
                    .bigram_right()
                    .iter()
                    .flatten()
                {
                    let right_fid = right_fid.get();
                    if bigram_fids.is_empty() {
                        bigram_fids.resize(1, HashMap::new());
                    }
                    let features = &mut bigram_fids[0];
                    if let RawEntryMut::Vacant(v) = features.raw_entry_mut().from_key(&right_fid) {
                        v.insert(right_fid, u32::try_from(weights.len()).unwrap());
                        weights.push(0.0);
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    fn update_features(
        lattice: &Lattice,
        provider: &FeatureProvider,
        unigram_fids: &mut Vec<u32>,
        bigram_fids: &mut Vec<HashMap<u32, u32>>,
        weights: &mut Vec<f64>,
    ) {
        for (i, node) in lattice.nodes().iter().enumerate() {
            if i == 0 {
                for curr_edge in node.edges() {
                    Self::update_bigram_feature(
                        provider,
                        None,
                        Some(curr_edge.label),
                        bigram_fids,
                        weights,
                    );
                }
            }
            for curr_edge in node.edges() {
                for next_edge in lattice.nodes()[curr_edge.target()].edges() {
                    Self::update_bigram_feature(
                        provider,
                        Some(curr_edge.label),
                        Some(next_edge.label),
                        bigram_fids,
                        weights,
                    );
                }
                if curr_edge.target() == lattice.nodes().len() - 1 {
                    Self::update_bigram_feature(
                        provider,
                        Some(curr_edge.label),
                        None,
                        bigram_fids,
                        weights,
                    );
                }
                Self::update_unigram_feature(provider, curr_edge.label, unigram_fids, weights);
            }
        }
    }

    /// Trains a model from the given dataset.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn train(&self, lattices: &[Lattice], mut provider: FeatureProvider) -> RawModel {
        let mut unigram_fids = vec![];
        let mut bigram_fids = vec![];
        let mut weights_init = vec![];

        for lattice in lattices {
            Self::update_features(
                lattice,
                &provider,
                &mut unigram_fids,
                &mut bigram_fids,
                &mut weights_init,
            );
        }

        let weights = sgd::solve(
            &lattices,
            &provider,
            &unigram_fids,
            &bigram_fids,
            &weights_init,
            self.regularization,
            self.lambda,
            self.max_iter,
            self.n_threads,
        );

        /*
        let weights = lbfgs::solve(
            &lattices,
            &provider,
            &unigram_fids,
            &bigram_fids,
            &weights_init,
            self.regularization,
            self.lambda,
            self.max_iter,
            self.n_threads,
        );
        */

        // Removes zero features
        let mut feature_id_map = HashMap::new();
        let mut new_weights = vec![0.0];
        for (i, w) in weights.into_iter().enumerate() {
            if w.abs() < f64::EPSILON {
                continue;
            }
            feature_id_map.insert(
                u32::try_from(i).unwrap(),
                u32::try_from(new_weights.len()).unwrap(),
            );
            new_weights.push(w);
        }
        let mut new_unigram_fids = vec![];
        for fid in unigram_fids {
            new_unigram_fids.push(feature_id_map.get(&fid).and_then(|&i| NonZeroU32::new(i)));
        }
        let mut new_bigram_fids = vec![];
        let mut right_id_used = HashSet::new();
        for fids in bigram_fids {
            let mut new_fids = HashMap::new();
            for (k, v) in fids {
                if let Some(&v) = feature_id_map.get(&v) {
                    new_fids.insert(k, v);
                    right_id_used.insert(k);
                }
            }
            new_bigram_fids.push(new_fids);
        }

        for feature_set in &mut provider.feature_sets {
            let mut new_unigram = vec![];
            for &fid in feature_set.unigram() {
                if new_unigram_fids
                    .get(usize::from_u32(fid.get() - 1))
                    .copied()
                    .flatten()
                    .is_some()
                {
                    new_unigram.push(fid);
                }
            }
            feature_set.unigram = new_unigram;
            for fid in &mut feature_set.bigram_left {
                *fid = fid.filter(|fid| {
                    !new_bigram_fids
                        .get(usize::from_u32(fid.get()))
                        .map_or(false, HashMap::is_empty)
                });
            }
            for fid in &mut feature_set.bigram_right {
                *fid = fid.filter(|fid| right_id_used.contains(&fid.get()));
            }
        }

        RawModel::new(new_weights, new_unigram_fids, new_bigram_fids, provider)
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}
