use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::lattice::Lattice;
use crate::math;

pub fn calculate_alphas_betas(
    lattice: &Lattice,
    weights: &[f64],
    unigram_fids: &[HashMap<usize, usize>],
    bigram_fids: &[HashMap<usize, usize>],
    alphas: &mut Vec<Vec<(usize, usize, f64)>>,
    betas: &mut Vec<Vec<(usize, usize, f64)>>,
) -> f64 {
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
    alphas[0].push((0, 0, 0.0));
    betas[lattice.nodes().len() - 1].push((0, 0, 0.0));

    // add 1-gram scores
    for (i, node) in lattice.nodes().iter().enumerate() {
        for edge in node.edges() {
            let mut score = 0.0;
            if let Some(unigram_fids) = unigram_fids.get(edge.label) {
                if let Some(features) = lattice.features().get(&(i, edge.target())) {
                    for feature in features {
                        if let Some(&fid) = unigram_fids.get(&feature.feature_id) {
                            score += weights[fid] * feature.value;
                        }
                    }
                }
            }
            alphas[edge.target()].push((i, edge.label, score));
            betas[i].push((edge.target(), edge.label, score));
        }
    }

    // alphas
    for i in 1..lattice.nodes().len() {
        for j in 0..alphas[i].len() {
            let (k, curr_label, _) = alphas[i][j];
            let mut score_total = f64::NEG_INFINITY;
            for &(_, prev_label, score) in &alphas[k] {
                if let Some(&fid) = bigram_fids[prev_label].get(&curr_label) {
                    score_total = math::logsumexp(score_total, score + weights[fid]);
                }
            }
            alphas[i][j].2 += score_total;
        }
    }

    // betas
    for i in (0..lattice.nodes().len() - 1).rev() {
        for j in 0..betas[i].len() {
            let (k, curr_label, _) = betas[i][j];
            let mut score_total = f64::NEG_INFINITY;
            for &(_, next_label, score) in &betas[k] {
                if let Some(&fid) = bigram_fids[curr_label].get(&next_label) {
                    score_total = math::logsumexp(score_total, score + weights[fid]);
                }
            }
            betas[i][j].2 += score_total;
        }
    }

    let mut score_total = f64::NEG_INFINITY;
    for &(_, next_label, score) in &betas[0] {
        let fid = *bigram_fids[0].get(&next_label).unwrap();
        score_total = math::logsumexp(score_total, score + weights[fid]);
    }
    score_total
}

pub fn calculate_loss(
    lattice: &Lattice,
    weights: &[f64],
    unigram_fids: &[HashMap<usize, usize>],
    bigram_fids: &[HashMap<usize, usize>],
    z: f64,
) -> f64 {
    let mut log_prob = z;
    let mut pos = 0;
    let mut prev_label = 0;
    while pos < lattice.nodes().len() - 1 {
        let edge = &lattice.nodes()[pos].edges()[0];
        if let Some(features) = lattice.features().get(&(pos, edge.target())) {
            for feature in features {
                if let Some(&fid) = unigram_fids
                    .get(edge.label)
                    .and_then(|hm| hm.get(&feature.feature_id))
                {
                    log_prob -= weights[fid] * feature.value;
                }
            }
        }
        if let Some(&fid) = bigram_fids
            .get(prev_label)
            .and_then(|hm| hm.get(&edge.label))
        {
            log_prob -= weights[fid];
        }
        pos = edge.target();
        prev_label = edge.label;
    }
    if let Some(&fid) = bigram_fids.get(prev_label).and_then(|hm| hm.get(&0)) {
        log_prob -= weights[fid];
    }
    log_prob
}

pub fn update_gradient(
    lattice: &Lattice,
    weights: &[f64],
    unigram_fids: &[HashMap<usize, usize>],
    bigram_fids: &[HashMap<usize, usize>],
    alphas: &[Vec<(usize, usize, f64)>],
    betas: &[Vec<(usize, usize, f64)>],
    z: f64,
    gradients: &mut [f64],
) {
    for pos in 0..lattice.nodes().len() {
        for &(prev_pos, prev_label, alpha) in &alphas[pos] {
            let mut prob_total = 0.0;
            for &(_, next_label, beta) in &betas[pos] {
                if let Some(&fid) = bigram_fids
                    .get(prev_label)
                    .and_then(|hm| hm.get(&next_label))
                {
                    let prob = (alpha + weights[fid] + beta - z).exp();
                    gradients[fid] += prob;
                    prob_total += prob;
                }
            }
            if let Some(features) = lattice.features().get(&(prev_pos, pos)) {
                for feature in features {
                    if let Some(&fid) = unigram_fids
                        .get(prev_label)
                        .and_then(|hm| hm.get(&feature.feature_id))
                    {
                        gradients[fid] += feature.value * prob_total;
                    }
                }
            }
        }
    }
    let mut pos = 0;
    let mut prev_label = 0;
    while pos < lattice.nodes().len() - 1 {
        let edge = &lattice.nodes()[pos].edges()[0];
        if let Some(features) = lattice.features().get(&(pos, edge.target())) {
            for feature in features {
                if let Some(&fid) = unigram_fids
                    .get(edge.label)
                    .and_then(|hm| hm.get(&feature.feature_id))
                {
                    gradients[fid] -= feature.value;
                }
            }
        }
        if let Some(&fid) = bigram_fids
            .get(prev_label)
            .and_then(|hm| hm.get(&edge.label))
        {
            gradients[fid] -= 1.0;
        }
        pos = edge.target();
        prev_label = edge.label;
    }
    if let Some(&fid) = bigram_fids.get(prev_label).and_then(|hm| hm.get(&0)) {
        gradients[fid] -= 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::num::NonZeroUsize;

    use crate::lattice::{Edge, Feature};
    use crate::test_utils::{assert_alpha_beta, hashmap};

    #[test]
    fn alpha_beta_test_1() {
        let mut lattice = Lattice::new(&[
            Edge::new(1, NonZeroUsize::new(3)),
            Edge::new(3, NonZeroUsize::new(4)),
            Edge::new(5, NonZeroUsize::new(2)),
            Edge::new(6, NonZeroUsize::new(2)),
        ]);
        lattice.add_feature(0, 1, Feature::new(4, 2.0));
        lattice.add_feature(5, 6, Feature::new(8, 3.0));

        let unigram_fids = vec![hashmap![], hashmap![], hashmap![8 => 6], hashmap![4 => 5]];
        let bigram_fids = vec![
            hashmap![3 => 0],
            hashmap![],
            hashmap![0 => 1, 2 => 2],
            hashmap![4 => 3],
            hashmap![2 => 4],
        ];
        let weights = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];

        let mut alphas = vec![];
        let mut betas = vec![];
        let z = calculate_alphas_betas(
            &lattice,
            &weights,
            &unigram_fids,
            &bigram_fids,
            &mut alphas,
            &mut betas,
        );

        let alphas_expected = vec![
            vec![(0, 0, 0.0)],
            vec![(0, 3, 26.0)],
            vec![],
            vec![(1, 4, 34.0)],
            vec![],
            vec![(3, 2, 44.0)],
            vec![(5, 2, 92.0)],
        ];
        assert_alpha_beta!(alphas_expected, alphas);

        let betas_expected = vec![
            vec![(1, 3, 94.0)],
            vec![(3, 4, 62.0)],
            vec![],
            vec![(5, 2, 52.0)],
            vec![],
            vec![(6, 2, 46.0)],
            vec![(0, 0, 0.0)],
        ];
        assert_alpha_beta!(betas_expected, betas);

        assert!((z - 96.0).abs() < f64::EPSILON);
    }

    #[test]
    fn alpha_beta_test_2() {
        let mut lattice = Lattice::new(&[
            Edge::new(1, NonZeroUsize::new(3)),
            Edge::new(3, NonZeroUsize::new(4)),
            Edge::new(5, NonZeroUsize::new(2)),
            Edge::new(6, NonZeroUsize::new(2)),
        ]);
        lattice.add_branch(0, Edge::new(3, NonZeroUsize::new(4)));
        lattice.add_branch(1, Edge::new(6, NonZeroUsize::new(4)));
        lattice.add_branch(5, Edge::new(6, NonZeroUsize::new(3)));
        lattice.add_feature(0, 1, Feature::new(4, 2.0));
        lattice.add_feature(5, 6, Feature::new(8, 3.0));

        let unigram_fids = vec![
            hashmap![],
            hashmap![],
            hashmap![8 => 6],
            hashmap![4 => 5, 8 => 7],
        ];
        let bigram_fids = vec![
            hashmap![3 => 0, 4 => 8],
            hashmap![],
            hashmap![0 => 1, 2 => 2, 3 => 9],
            hashmap![0 => 10, 4 => 3],
            hashmap![0 => 11, 2 => 4],
        ];
        let weights = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        ];

        let mut alphas = vec![];
        let mut betas = vec![];
        let z = calculate_alphas_betas(
            &lattice,
            &weights,
            &unigram_fids,
            &bigram_fids,
            &mut alphas,
            &mut betas,
        );

        let alphas_expected = vec![
            vec![(0, 0, 0.0)],
            vec![(0, 3, 26.0)],
            vec![],
            vec![(0, 4, 18.0), (1, 4, 34.0)],
            vec![],
            vec![(3, 2, math::logsumexp(28.0, 44.0))],
            vec![
                (1, 4, 34.0),
                (5, 2, math::logsumexp(28.0, 44.0) + 48.0),
                (5, 3, math::logsumexp(28.0, 44.0) + 68.0),
            ],
        ];
        assert_alpha_beta!(alphas_expected, alphas);

        let betas_expected = vec![
            vec![
                (
                    1,
                    3,
                    math::logsumexp(32.0, math::logsumexp(52.0, 90.0) + 18.0) + 24.0,
                ),
                (3, 4, math::logsumexp(52.0, 90.0) + 10.0),
            ],
            vec![(3, 4, math::logsumexp(52.0, 90.0) + 10.0), (6, 4, 24.0)],
            vec![],
            vec![(5, 2, math::logsumexp(52.0, 90.0))],
            vec![],
            vec![(6, 2, 46.0), (6, 3, 70.0)],
            vec![(0, 0, 0.0)],
        ];
        assert_alpha_beta!(betas_expected, betas);

        assert!(
            (math::logsumexp(
                math::logsumexp(32.0, math::logsumexp(52.0, 90.0) + 18.0) + 26.0,
                math::logsumexp(52.0, 90.0) + 28.0
            ) - z)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn probability_test() {
        let mut lattice = Lattice::new(&[
            Edge::new(1, NonZeroUsize::new(3)),
            Edge::new(3, NonZeroUsize::new(4)),
            Edge::new(5, NonZeroUsize::new(2)),
            Edge::new(6, NonZeroUsize::new(2)),
        ]);
        lattice.add_branch(0, Edge::new(3, NonZeroUsize::new(4)));
        lattice.add_branch(1, Edge::new(6, NonZeroUsize::new(4)));
        lattice.add_branch(5, Edge::new(6, NonZeroUsize::new(3)));
        lattice.add_feature(0, 1, Feature::new(4, 2.0));
        lattice.add_feature(5, 6, Feature::new(8, 3.0));

        let unigram_fids = vec![
            hashmap![],
            hashmap![],
            hashmap![8 => 6],
            hashmap![4 => 5, 8 => 7],
        ];
        let bigram_fids = vec![
            hashmap![3 => 0, 4 => 8],
            hashmap![],
            hashmap![0 => 1, 2 => 2, 3 => 9],
            hashmap![0 => 10, 4 => 3],
            hashmap![0 => 11, 2 => 4],
        ];
        let weights = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        ];

        let loss = calculate_loss(&lattice, &weights, &unigram_fids, &bigram_fids, 100.0);

        assert!((96.0 - 100.0 + loss).abs() < f64::EPSILON);
    }
}
