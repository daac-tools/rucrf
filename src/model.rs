use alloc::vec::Vec;

use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    Decode, Encode,
};
use hashbrown::HashMap;

use crate::lattice::{Edge, Lattice};

/// Represents a model of CRF
pub struct Model {
    /// Weight vector
    pub weights: Vec<f64>,

    /// Map of uni-gram feature IDs and weight indices
    pub unigram_fids: Vec<HashMap<usize, usize>>,

    /// Map of bi-gram feature IDs and weight indices
    pub bigram_fids: Vec<HashMap<usize, usize>>,
}

impl Model {
    /// Searches the best path of the given lattice.
    pub fn search_best_path(&self, lattice: &Lattice) -> Vec<Edge> {
        let mut best_scores = vec![vec![]; lattice.nodes().len()];
        best_scores[lattice.nodes().len() - 1].push((0, 0, 0, 0.0));
        for (i, node) in lattice.nodes().iter().enumerate() {
            for edge in node.edges() {
                let mut score = 0.0;
                if let Some(unigram_fids) = self.unigram_fids.get(edge.label) {
                    if let Some(features) = lattice.features().get(&(i, edge.target())) {
                        for feature in features {
                            if let Some(&fid) = unigram_fids.get(&feature.feature_id) {
                                score += self.weights[fid] * feature.value;
                            }
                        }
                    }
                }
                best_scores[i].push((edge.target(), 0, edge.label, score));
            }
        }
        for i in (0..lattice.nodes().len() - 1).rev() {
            for j in 0..best_scores[i].len() {
                let (k, _, curr_label, _) = best_scores[i][j];
                let mut best_score = f64::NEG_INFINITY;
                let mut best_idx = 0;
                for (p, &(_, _, next_label, score)) in best_scores[k].iter().enumerate() {
                    let score = score
                        + if let Some(&fid) = self
                            .bigram_fids
                            .get(curr_label)
                            .and_then(|hm| hm.get(&next_label))
                        {
                            self.weights[fid]
                        } else {
                            0.0
                        };
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
        for (p, &(_, _, next_label, score)) in best_scores[0].iter().enumerate() {
            let score = score
                + if let Some(&fid) = self.bigram_fids[0].get(&next_label) {
                    self.weights[fid]
                } else {
                    0.0
                };
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
        best_path
    }
}

impl Decode for Model {
    #[allow(clippy::type_complexity)]
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let weights = Decode::decode(decoder)?;
        let unigram_fids: Vec<Vec<(usize, usize)>> = Decode::decode(decoder)?;
        let bigram_fids: Vec<Vec<(usize, usize)>> = Decode::decode(decoder)?;
        Ok(Self {
            weights,
            unigram_fids: unigram_fids
                .into_iter()
                .map(|v| v.into_iter().collect())
                .collect(),
            bigram_fids: bigram_fids
                .into_iter()
                .map(|v| v.into_iter().collect())
                .collect(),
        })
    }
}

impl Encode for Model {
    #[allow(clippy::type_complexity)]
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let unigram_fids: Vec<Vec<(usize, usize)>> = self
            .unigram_fids
            .iter()
            .map(|v| v.iter().map(|(&k, &v)| (k, v)).collect())
            .collect();
        let bigram_fids: Vec<Vec<(usize, usize)>> = self
            .bigram_fids
            .iter()
            .map(|v| v.iter().map(|(&k, &v)| (k, v)).collect())
            .collect();
        Encode::encode(&self.weights, encoder)?;
        Encode::encode(&unigram_fids, encoder)?;
        Encode::encode(&bigram_fids, encoder)?;
        Ok(())
    }
}
