use core::num::NonZeroU32;

use alloc::vec::Vec;

use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    Decode, Encode,
};
use hashbrown::HashMap;

use crate::errors::{Result, RucrfError};
use crate::feature::{self, FeatureProvider};
use crate::lattice::{Edge, Lattice};
use crate::utils::FromU32;

/// The `Model` trait allows for searching the best path in the lattice.
pub trait Model {
    /// Searches the best path and returns the path and its score.
    fn search_best_path(&self, lattice: &Lattice) -> (Vec<Edge>, f64);
}

/// Represents a raw model.
pub struct RawModel {
    weights: Vec<f64>,
    unigram_fids: Vec<Option<NonZeroU32>>,
    bigram_fids: Vec<HashMap<u32, u32>>,
    provider: FeatureProvider,
}

impl Decode for RawModel {
    #[allow(clippy::type_complexity)]
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let weights = Decode::decode(decoder)?;
        let unigram_fids: Vec<Option<NonZeroU32>> = Decode::decode(decoder)?;
        let bigram_fids: Vec<Vec<(u32, u32)>> = Decode::decode(decoder)?;
        let provider: FeatureProvider = Decode::decode(decoder)?;
        Ok(Self {
            weights,
            unigram_fids,
            bigram_fids: bigram_fids
                .into_iter()
                .map(|v| v.into_iter().collect())
                .collect(),
            provider,
        })
    }
}

impl Encode for RawModel {
    #[allow(clippy::type_complexity)]
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let bigram_fids: Vec<Vec<(u32, u32)>> = self
            .bigram_fids
            .iter()
            .map(|v| v.iter().map(|(&k, &v)| (k, v)).collect())
            .collect();
        Encode::encode(&self.weights, encoder)?;
        Encode::encode(&self.unigram_fids, encoder)?;
        Encode::encode(&bigram_fids, encoder)?;
        Encode::encode(&self.provider, encoder)?;
        Ok(())
    }
}

impl RawModel {
    #[cfg(feature = "train")]
    pub(crate) fn new(
        weights: Vec<f64>,
        unigram_fids: Vec<Option<NonZeroU32>>,
        bigram_fids: Vec<HashMap<u32, u32>>,
        provider: FeatureProvider,
    ) -> Self {
        Self {
            weights,
            unigram_fids,
            bigram_fids,
            provider,
        }
    }

    /// Returns a mutable reference of the feature provider.
    pub fn feature_provider(&mut self) -> &mut FeatureProvider {
        &mut self.provider
    }

    /// Merges this model and returns [`MergedModel`].
    ///
    /// This process integrates the features, so that each edge has three items: a uni-gram cost,
    /// a left-connection ID, and a right-connection ID.
    ///
    /// # Errors
    ///
    /// Generated left/right connection ID must be smaller than 2^32.
    pub fn merge(&self) -> Result<MergedModel> {
        let mut left_conn_ids = HashMap::new();
        let mut right_conn_ids = HashMap::new();
        let mut left_conn_to_right_feats = vec![];
        let mut right_conn_to_left_feats = vec![];
        let mut new_feature_sets = vec![];
        for feature_set in &self.provider.feature_sets {
            let mut weight = 0.0;
            for fid in feature_set.unigram() {
                let fid = usize::from_u32(fid.get() - 1);
                if let Some(fid) = self.unigram_fids.get(fid).copied().flatten() {
                    let fid = usize::from_u32(fid.get());
                    weight += self.weights[fid];
                }
            }
            let left_id = {
                let new_id = u32::try_from(left_conn_to_right_feats.len() + 1)
                    .map_err(|_| RucrfError::model_scale("connection ID too large"))?;
                *left_conn_ids
                    .raw_entry_mut()
                    .from_key(feature_set.bigram_right())
                    .or_insert_with(|| {
                        let features = feature_set.bigram_right().to_vec();
                        left_conn_to_right_feats.push(features.clone());
                        // Safety: new_id is always greater than or equal to 1.
                        (features, unsafe { NonZeroU32::new_unchecked(new_id) })
                    })
                    .1
            };
            let right_id = {
                let new_id = u32::try_from(right_conn_to_left_feats.len() + 1)
                    .map_err(|_| RucrfError::model_scale("connection ID too large"))?;
                *right_conn_ids
                    .raw_entry_mut()
                    .from_key(feature_set.bigram_left())
                    .or_insert_with(|| {
                        let features = feature_set.bigram_left().to_vec();
                        right_conn_to_left_feats.push(features.clone());
                        // Safety: new_id is always greater than or equal to 1.
                        (features, unsafe { NonZeroU32::new_unchecked(new_id) })
                    })
                    .1
            };
            new_feature_sets.push(MergedFeatureSet {
                weight,
                left_id,
                right_id,
            });
        }
        let mut matrix = vec![];

        // BOS
        let mut m = HashMap::new();
        for (i, left_ids) in left_conn_to_right_feats.iter().enumerate() {
            let mut weight = 0.0;
            for fid in left_ids.iter().flatten() {
                if let Some(&fid) = self.bigram_fids[0].get(&fid.get()) {
                    weight += self.weights[usize::from_u32(fid)];
                }
            }
            if weight.abs() >= f64::EPSILON {
                m.insert(
                    u32::try_from(i + 1)
                        .map_err(|_| RucrfError::model_scale("connection ID too large"))?,
                    weight,
                );
            }
        }
        matrix.push(m);

        for right_ids in &right_conn_to_left_feats {
            let mut m = HashMap::new();

            // EOS
            let mut weight = 0.0;
            for fid in right_ids.iter().flatten() {
                let right_id = usize::from_u32(fid.get());
                if let Some(&fid) = self.bigram_fids.get(right_id).and_then(|hm| hm.get(&0)) {
                    weight += self.weights[usize::from_u32(fid)];
                }
            }
            if weight.abs() >= f64::EPSILON {
                m.insert(0, weight);
            }

            for (i, left_ids) in left_conn_to_right_feats.iter().enumerate() {
                let mut weight = 0.0;
                for (right_id, left_id) in right_ids.iter().zip(left_ids) {
                    if let (Some(right_id), Some(left_id)) = (right_id, left_id) {
                        let right_id = usize::from_u32(right_id.get());
                        let left_id = left_id.get();
                        if let Some(&fid) = self
                            .bigram_fids
                            .get(right_id)
                            .and_then(|hm| hm.get(&left_id))
                        {
                            weight += self.weights[usize::from_u32(fid)];
                        }
                    }
                }
                if weight.abs() >= f64::EPSILON {
                    m.insert(
                        u32::try_from(i + 1)
                            .map_err(|_| RucrfError::model_scale("connection ID too large"))?,
                        weight,
                    );
                }
            }

            matrix.push(m);
        }

        Ok(MergedModel {
            feature_sets: new_feature_sets,
            matrix,
            left_conn_to_right_feats,
            right_conn_to_left_feats,
        })
    }

    /// Returns the relation between uni-gram feature IDs and weight indices.
    #[must_use]
    pub fn unigram_feature_ids(&self) -> &[Option<NonZeroU32>] {
        &self.unigram_fids
    }

    /// Returns the relation between bi-gram feature IDs and weight indices.
    #[must_use]
    pub fn bigram_feature_ids(&self) -> &[HashMap<u32, u32>] {
        &self.bigram_fids
    }

    /// Returns weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

impl Model for RawModel {
    #[must_use]
    fn search_best_path(&self, lattice: &Lattice) -> (Vec<Edge>, f64) {
        let mut best_scores = vec![vec![]; lattice.nodes().len()];
        best_scores[lattice.nodes().len() - 1].push((0, 0, None, 0.0));
        for (i, node) in lattice.nodes().iter().enumerate() {
            for edge in node.edges() {
                let mut score = 0.0;
                if let Some(feature_set) = self.provider.get_feature_set(edge.label) {
                    for &fid in feature_set.unigram() {
                        let fid = usize::from_u32(fid.get() - 1);
                        if let Some(fid) = self.unigram_fids[fid] {
                            let fid = usize::from_u32(fid.get());
                            score += self.weights[fid];
                        }
                    }
                }
                best_scores[i].push((edge.target(), 0, Some(edge.label), score));
            }
        }
        for i in (0..lattice.nodes().len() - 1).rev() {
            for j in 0..best_scores[i].len() {
                let (k, _, curr_label, _) = best_scores[i][j];
                let mut best_score = f64::NEG_INFINITY;
                let mut best_idx = 0;
                for (p, &(_, _, next_label, mut score)) in best_scores[k].iter().enumerate() {
                    feature::apply_bigram(
                        curr_label,
                        next_label,
                        &self.provider,
                        &self.bigram_fids,
                        |fid| {
                            score += self.weights[usize::from_u32(fid)];
                        },
                    );
                    if score > best_score {
                        best_score = score;
                        best_idx = p;
                    }
                }
                best_scores[i][j].1 = best_idx;
                best_scores[i][j].3 += best_score;
            }
        }
        let mut best_score = f64::NEG_INFINITY;
        let mut idx = 0;
        for (p, &(_, _, next_label, mut score)) in best_scores[0].iter().enumerate() {
            feature::apply_bigram(None, next_label, &self.provider, &self.bigram_fids, |fid| {
                score += self.weights[usize::from_u32(fid)];
            });
            if score > best_score {
                best_score = score;
                idx = p;
            }
        }
        let mut pos = 0;
        let mut best_path = vec![];
        while pos < lattice.nodes().len() - 1 {
            let edge = &lattice.nodes()[pos].edges()[idx];
            idx = best_scores[pos][idx].1;
            pos = edge.target();
            best_path.push(Edge::new(pos, edge.label()));
        }
        (best_path, best_score)
    }
}

/// Represents a merged feature set.
#[derive(Clone, Copy, Debug)]
pub struct MergedFeatureSet {
    /// Weight.
    pub weight: f64,
    /// Left bi-gram connection ID.
    pub left_id: NonZeroU32,
    /// Right bi-gram connection ID.
    pub right_id: NonZeroU32,
}

/// Represents a merged model.
pub struct MergedModel {
    /// Feature sets corresponding to label IDs.
    pub feature_sets: Vec<MergedFeatureSet>,
    /// Bi-gram weight matrix.
    pub matrix: Vec<HashMap<u32, f64>>,
    /// Relation between the left connection IDs and the right bi-gram feature IDs.
    pub left_conn_to_right_feats: Vec<Vec<Option<NonZeroU32>>>,
    /// Relation between the right connection IDs and the left bi-gram feature IDs.
    pub right_conn_to_left_feats: Vec<Vec<Option<NonZeroU32>>>,
}

impl Model for MergedModel {
    #[must_use]
    fn search_best_path(&self, lattice: &Lattice) -> (Vec<Edge>, f64) {
        let mut best_scores = vec![vec![]; lattice.nodes().len()];
        best_scores[lattice.nodes().len() - 1].push((0, 0, None, 0.0));
        for (i, node) in lattice.nodes().iter().enumerate() {
            for edge in node.edges() {
                let label = usize::from_u32(edge.label.get() - 1);
                let score = self.feature_sets.get(label).map_or(0.0, |s| s.weight);
                best_scores[i].push((edge.target(), 0, Some(edge.label), score));
            }
        }
        for i in (0..lattice.nodes().len() - 1).rev() {
            for j in 0..best_scores[i].len() {
                let (k, _, curr_label, _) = best_scores[i][j];
                let mut best_score = f64::NEG_INFINITY;
                let mut best_idx = 0;
                let curr_id = curr_label.map_or(Some(0), |label| {
                    self.feature_sets
                        .get(usize::from_u32(label.get() - 1))
                        .map(|s| s.right_id.get())
                });
                for (p, &(_, _, next_label, mut score)) in best_scores[k].iter().enumerate() {
                    let next_id = next_label.map_or(Some(0), |label| {
                        self.feature_sets
                            .get(usize::from_u32(label.get() - 1))
                            .map(|s| s.left_id.get())
                    });
                    if let (Some(curr_id), Some(next_id)) = (curr_id, next_id) {
                        score += self
                            .matrix
                            .get(usize::from_u32(curr_id))
                            .and_then(|hm| hm.get(&next_id))
                            .unwrap_or(&0.0);
                    }
                    if score > best_score {
                        best_score = score;
                        best_idx = p;
                    }
                }
                best_scores[i][j].1 = best_idx;
                best_scores[i][j].3 += best_score;
            }
        }
        let mut best_score = f64::NEG_INFINITY;
        let mut idx = 0;
        for (p, &(_, _, next_label, mut score)) in best_scores[0].iter().enumerate() {
            let next_id = next_label.map_or(Some(0), |label| {
                self.feature_sets
                    .get(usize::from_u32(label.get() - 1))
                    .map(|s| s.right_id.get())
            });
            if let Some(next_id) = next_id {
                score += self
                    .matrix
                    .get(0)
                    .and_then(|hm| hm.get(&next_id))
                    .unwrap_or(&0.0);
            }
            if score > best_score {
                best_score = score;
                idx = p;
            }
        }
        let mut pos = 0;
        let mut best_path = vec![];
        while pos < lattice.nodes().len() - 1 {
            let edge = &lattice.nodes()[pos].edges()[idx];
            idx = best_scores[pos][idx].1;
            pos = edge.target();
            best_path.push(Edge::new(pos, edge.label()));
        }
        (best_path, best_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::num::NonZeroU32;

    use crate::lattice::Edge;
    use crate::test_utils::{self, hashmap};

    #[test]
    fn test_search_best_path() {
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
        // 5->3: 38 (2-2:16 3-3:22)
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
        let model = RawModel {
            weights: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 13.0, 24.0, 5.0, 26.0, 27.0, 6.0,
            ],
            unigram_fids: vec![
                NonZeroU32::new(1),
                NonZeroU32::new(3),
                NonZeroU32::new(5),
                NonZeroU32::new(7),
            ],
            bigram_fids: vec![
                hashmap![0 => 28, 1 => 0, 2 => 2, 3 => 4, 4 => 6],
                hashmap![0 => 8, 1 => 9, 2 => 10, 3 => 11, 4 => 12],
                hashmap![0 => 13, 1 => 14, 2 => 15, 3 => 16, 4 => 17],
                hashmap![0 => 18, 1 => 19, 2 => 20, 3 => 21, 4 => 22],
                hashmap![0 => 23, 1 => 24, 2 => 25, 3 => 26, 4 => 27],
            ],
            provider: test_utils::generate_test_feature_provider(),
        };
        let lattice = test_utils::generate_test_lattice();

        let (path, score) = model.search_best_path(&lattice);

        assert_eq!(
            vec![
                Edge::new(1, NonZeroU32::new(1).unwrap()),
                Edge::new(2, NonZeroU32::new(2).unwrap()),
                Edge::new(3, NonZeroU32::new(6).unwrap()),
                Edge::new(4, NonZeroU32::new(7).unwrap()),
                Edge::new(5, NonZeroU32::new(4).unwrap()),
            ],
            path,
        );
        assert!((194.0 - score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hashed_search_best_path() {
        let model = RawModel {
            weights: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 13.0, 24.0, 5.0, 26.0, 27.0, 6.0,
            ],
            unigram_fids: vec![
                NonZeroU32::new(1),
                NonZeroU32::new(3),
                NonZeroU32::new(5),
                NonZeroU32::new(7),
            ],
            bigram_fids: vec![
                hashmap![0 => 28, 1 => 0, 2 => 2, 3 => 4, 4 => 6],
                hashmap![0 => 8, 1 => 9, 2 => 10, 3 => 11, 4 => 12],
                hashmap![0 => 13, 1 => 14, 2 => 15, 3 => 16, 4 => 17],
                hashmap![0 => 18, 1 => 19, 2 => 20, 3 => 21, 4 => 22],
                hashmap![0 => 23, 1 => 24, 2 => 25, 3 => 26, 4 => 27],
            ],
            provider: test_utils::generate_test_feature_provider(),
        };
        let compiled_model = model.merge().unwrap();

        let lattice = test_utils::generate_test_lattice();

        let (path, score) = compiled_model.search_best_path(&lattice);

        assert_eq!(
            vec![
                Edge::new(1, NonZeroU32::new(1).unwrap()),
                Edge::new(2, NonZeroU32::new(2).unwrap()),
                Edge::new(3, NonZeroU32::new(6).unwrap()),
                Edge::new(4, NonZeroU32::new(7).unwrap()),
                Edge::new(5, NonZeroU32::new(4).unwrap()),
            ],
            path,
        );
        assert!((194.0 - score).abs() < f64::EPSILON);
    }
}
