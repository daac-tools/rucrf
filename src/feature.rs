use core::num::NonZeroU32;

use alloc::vec::Vec;

use bincode::{Decode, Encode};
use hashbrown::HashMap;

use crate::utils::FromU32;

#[inline(always)]
pub fn apply_bigram<F>(
    left_label: Option<NonZeroU32>,
    right_label: Option<NonZeroU32>,
    provider: &FeatureProvider,
    bigram_fids: &[HashMap<u32, u32>],
    mut f: F,
) where
    F: FnMut(u32),
{
    match (left_label, right_label) {
        (Some(left_label), Some(right_label)) => {
            for (&left_fid, &right_fid) in provider
                .get_feature_set(left_label)
                .left()
                .iter()
                .zip(provider.get_feature_set(right_label).right())
            {
                if let (Some(left_fid), Some(right_fid)) = (left_fid, right_fid) {
                    let left_fid = usize::from_u32(left_fid.get());
                    let right_fid = right_fid.get();
                    if let Some(&fid) = bigram_fids.get(left_fid).and_then(|hm| hm.get(&right_fid))
                    {
                        f(fid);
                    }
                }
            }
        }
        (Some(left_label), None) => {
            for &left_fid in provider.get_feature_set(left_label).left() {
                if let Some(left_fid) = left_fid {
                    let left_fid = usize::from_u32(left_fid.get());
                    if let Some(&fid) = bigram_fids[left_fid].get(&0) {
                        f(fid);
                    }
                }
            }
        }
        (None, Some(right_label)) => {
            for &right_fid in provider.get_feature_set(right_label).right() {
                if let Some(right_fid) = right_fid {
                    let right_fid = right_fid.get();
                    if let Some(&fid) = bigram_fids[0].get(&right_fid) {
                        f(fid);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Manages a set of features for each label.
#[derive(Debug, Default, Decode, Encode)]
pub struct FeatureSet {
    pub(crate) unigram: Vec<NonZeroU32>,
    pub(crate) left: Vec<Option<NonZeroU32>>,
    pub(crate) right: Vec<Option<NonZeroU32>>,
}

impl FeatureSet {
    /// Creates a new [`FeatureSet`].
    #[inline(always)]
    #[must_use]
    pub fn new(
        unigram: &[NonZeroU32],
        left: &[Option<NonZeroU32>],
        right: &[Option<NonZeroU32>],
    ) -> Self {
        Self {
            unigram: unigram.to_vec(),
            left: left.to_vec(),
            right: right.to_vec(),
        }
    }

    /// Gets uni-gram feature IDs.
    #[inline(always)]
    #[must_use]
    pub fn unigram(&self) -> &[NonZeroU32] {
        &self.unigram
    }

    /// Gets left bi-gram feature IDs
    #[inline(always)]
    #[must_use]
    pub fn left(&self) -> &[Option<NonZeroU32>] {
        &self.left
    }

    /// Gets right bi-gram feature IDs.
    #[inline(always)]
    #[must_use]
    pub fn right(&self) -> &[Option<NonZeroU32>] {
        &self.right
    }
}

/// Manages the correspondence between edge labels and feature IDs.
#[derive(Debug, Default, Decode, Encode)]
pub struct FeatureProvider {
    pub(crate) feature_sets: Vec<FeatureSet>,
}

impl FeatureProvider {
    /// Creates a new [`FeatureProvider`].
    #[inline(always)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the manager has no item.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.feature_sets.is_empty()
    }

    /// Returns the number of items.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.feature_sets.len()
    }

    /// Adds a feature set and returns its ID.
    ///
    /// # Panics
    ///
    /// The number of features must be less than 2^32 - 1.
    #[inline(always)]
    pub fn add_feature_set(&mut self, feature_set: FeatureSet) -> NonZeroU32 {
        self.feature_sets.push(feature_set);
        NonZeroU32::new(u32::try_from(self.feature_sets.len()).unwrap()).unwrap()
    }

    /// Returns the reference to the feature set corresponding to the given ID.
    #[inline(always)]
    pub(crate) fn get_feature_set(&self, label: NonZeroU32) -> &FeatureSet {
        &self.feature_sets[usize::try_from(label.get() - 1).unwrap()]
    }
}
