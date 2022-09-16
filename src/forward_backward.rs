use core::num::NonZeroU32;

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::feature::apply_bigram;
use crate::feature::FeatureProvider;
use crate::lattice::Lattice;
use crate::math;
use crate::utils::FromU32;
use crate::vector::{GradientVector, WeightVector};

pub fn calculate_alphas_betas<W>(
    lattice: &Lattice,
    provider: &FeatureProvider,
    weights: &W,
    unigram_fids: &[u32],
    bigram_fids: &[HashMap<u32, u32>],
    alphas: &mut Vec<Vec<(usize, Option<NonZeroU32>, f64)>>,
    betas: &mut Vec<Vec<(usize, Option<NonZeroU32>, f64)>>,
) -> f64
where
    W: WeightVector,
{
    if alphas.len() < lattice.nodes().len() {
        alphas.resize(lattice.nodes().len(), vec![]);
        betas.resize(lattice.nodes().len(), vec![]);
    }
    for x in &mut alphas[..lattice.nodes().len()] {
        x.clear();
    }
    for x in &mut betas[..lattice.nodes().len()] {
        x.clear();
    }
    alphas[0].push((0, None, 0.0));
    betas[lattice.nodes().len() - 1].push((0, None, 0.0));

    // add 1-gram scores
    for (i, (node, betas)) in lattice.nodes().iter().zip(betas.iter_mut()).enumerate() {
        for edge in node.edges() {
            let mut score = 0.0;
            for &fid in provider.get_feature_set(edge.label).unigram() {
                let fid = usize::try_from(fid.get() - 1).unwrap();
                let fid = unigram_fids[fid];
                score += weights.get_weight(usize::from_u32(fid));
            }
            alphas[edge.target()].push((i, Some(edge.label), score));
            betas.push((edge.target(), Some(edge.label), score));
        }
    }

    // alphas
    for i in 1..lattice.nodes().len() {
        for j in 0..alphas[i].len() {
            let (k, curr_label, _) = alphas[i][j];
            let mut score_total = f64::NEG_INFINITY;
            for &(_, prev_label, mut score) in &alphas[k] {
                apply_bigram(prev_label, curr_label, provider, bigram_fids, |fid| {
                    score += weights.get_weight(usize::from_u32(fid));
                });
                score_total = math::logsumexp(score_total, score);
            }
            alphas[i][j].2 += score_total;
        }
    }

    // betas
    for i in (0..lattice.nodes().len() - 1).rev() {
        for j in 0..betas[i].len() {
            let (k, curr_label, _) = betas[i][j];
            let mut score_total = f64::NEG_INFINITY;
            for &(_, next_label, mut score) in &betas[k] {
                apply_bigram(curr_label, next_label, provider, bigram_fids, |fid| {
                    score += weights.get_weight(usize::from_u32(fid));
                });
                score_total = math::logsumexp(score_total, score);
            }
            betas[i][j].2 += score_total;
        }
    }

    let mut score_total = f64::NEG_INFINITY;
    for &(_, next_label, mut score) in &betas[0] {
        apply_bigram(None, next_label, provider, bigram_fids, |fid| {
            score += weights.get_weight(usize::from_u32(fid));
        });
        score_total = math::logsumexp(score_total, score);
    }
    score_total
}

pub fn calculate_loss<W>(
    lattice: &Lattice,
    provider: &FeatureProvider,
    weights: &W,
    unigram_fids: &[u32],
    bigram_fids: &[HashMap<u32, u32>],
    z: f64,
) -> f64
where
    W: WeightVector,
{
    let mut log_prob = z;
    let mut pos = 0;
    let mut prev_label = None;

    while pos < lattice.nodes().len() - 1 {
        let edge = &lattice.nodes()[pos].edges()[0];
        for &fid in provider.get_feature_set(edge.label).unigram() {
            let fid = usize::from_u32(fid.get() - 1);
            let fid = usize::from_u32(unigram_fids[fid]);
            log_prob -= weights.get_weight(fid);
        }
        apply_bigram(prev_label, Some(edge.label), provider, bigram_fids, |fid| {
            log_prob -= weights.get_weight(usize::from_u32(fid));
        });
        pos = edge.target();
        prev_label = Some(edge.label);
    }
    for &prev_fid in provider.get_feature_set(prev_label.unwrap()).bigram_left() {
        if let Some(prev_fid) = prev_fid {
            let prev_fid = usize::try_from(prev_fid.get()).unwrap();
            if let Some(&fid) = bigram_fids[prev_fid].get(&0) {
                log_prob -= weights.get_weight(usize::from_u32(fid));
            }
        }
    }
    log_prob
}

#[allow(clippy::too_many_arguments)]
pub fn update_gradient<W, G>(
    lattice: &Lattice,
    provider: &FeatureProvider,
    weights: &W,
    unigram_fids: &[u32],
    bigram_fids: &[HashMap<u32, u32>],
    alphas: &[Vec<(usize, Option<NonZeroU32>, f64)>],
    betas: &[Vec<(usize, Option<NonZeroU32>, f64)>],
    z: f64,
    gradients: &mut G,
) where
    W: WeightVector,
    G: GradientVector,
{
    for (alphas, betas) in alphas.iter().zip(betas).take(lattice.nodes().len()) {
        for &(_, prev_label, alpha) in alphas {
            let mut prob_total = 0.0;
            for &(_, next_label, beta) in betas {
                let mut log_prob = alpha + beta - z;
                apply_bigram(prev_label, next_label, provider, bigram_fids, |fid| {
                    log_prob += weights.get_weight(usize::from_u32(fid));
                });
                let prob = log_prob.exp();
                prob_total += prob;
                apply_bigram(prev_label, next_label, provider, bigram_fids, |fid| {
                    gradients.add(usize::from_u32(fid), prob);
                });
            }
            if let Some(prev_label) = prev_label {
                for &fid in provider.get_feature_set(prev_label).unigram() {
                    let fid = usize::try_from(fid.get() - 1).unwrap();
                    let fid = usize::from_u32(unigram_fids[fid]);
                    gradients.add(fid, prob_total);
                }
            }
        }
    }
    let mut pos = 0;
    let mut prev_label = None;
    while pos < lattice.nodes().len() - 1 {
        let edge = &lattice.nodes()[pos].edges()[0];
        for &fid in provider.get_feature_set(edge.label).unigram() {
            let fid = usize::try_from(fid.get() - 1).unwrap();
            let fid = usize::from_u32(unigram_fids[fid]);
            gradients.add(fid, -1.0);
        }
        apply_bigram(prev_label, Some(edge.label), provider, bigram_fids, |fid| {
            gradients.add(usize::from_u32(fid), -1.0);
        });
        pos = edge.target();
        prev_label = Some(edge.label);
    }
    apply_bigram(prev_label, None, provider, bigram_fids, |fid| {
        gradients.add(usize::from_u32(fid), -1.0);
    });
}
