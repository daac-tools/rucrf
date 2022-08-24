use std::num::NonZeroUsize;

use daachorse::DoubleArrayAhoCorasick;
use hashbrown::{HashMap, HashSet};
use rucrf::{Edge, Lattice, Regularization, Trainer};
use vaporetto::Sentence;

use crate::feature_extractor::FeatureExtractor;
use crate::MAModel;

pub fn train(sentences: &[Sentence], dict: &[Sentence], n_threads: usize) -> MAModel {
    let trainer = Trainer::new()
        .n_threads(n_threads)
        .unwrap()
        .regularization(Regularization::L1, 0.1)
        .unwrap()
        .max_iter(300)
        .unwrap();

    let mut surf_label_map = HashMap::new();
    let mut label_tag_map = vec![];
    let mut tag_label_map = HashMap::new();

    for s in sentences {
        for token in s.iter_tokens() {
            let tag = token
                .tags()
                .get(0)
                .and_then(|tag| tag.as_ref().map(|tag| tag.as_ref()))
                .unwrap_or("");
            let label = *tag_label_map.entry(tag).or_insert(label_tag_map.len() + 1);
            if label == label_tag_map.len() + 1 {
                label_tag_map.push(tag.to_string());
            }
            surf_label_map
                .entry(token.surface())
                .or_insert_with(|| HashSet::new())
                .insert(label);
        }
    }
    for s in dict {
        for token in s.iter_tokens() {
            let tag = token
                .tags()
                .get(0)
                .and_then(|tag| tag.as_ref().map(|tag| tag.as_ref()))
                .unwrap_or("");
            let label = *tag_label_map.entry(tag).or_insert(label_tag_map.len() + 1);
            if label == label_tag_map.len() + 1 {
                label_tag_map.push(tag.to_string());
            }
            surf_label_map
                .entry(token.surface())
                .or_insert_with(|| HashSet::new())
                .insert(label);
        }
    }

    let mut surfaces = vec![];
    let mut labels = vec![];
    for (s, ls) in &surf_label_map {
        surfaces.push(s.to_string());
        labels.push(ls.iter().cloned().collect::<Vec<usize>>());
    }
    let pma = DoubleArrayAhoCorasick::<usize>::new(&surfaces).unwrap();

    let mut feature_extractor = FeatureExtractor::new(3);

    let mut lattices = vec![];

    for s in sentences {
        let mut edges = vec![];
        let mut trunk_labels = HashMap::new();
        for token in s.iter_tokens() {
            let tag = token
                .tags()
                .get(0)
                .and_then(|tag| tag.as_ref().map(|tag| tag.as_ref()))
                .unwrap_or("");
            let label = *tag_label_map.get(tag).unwrap();
            edges.push(Edge::new(token.end(), NonZeroUsize::new(label)));
            trunk_labels.insert((token.start(), token.end()), label);
        }
        let mut lattice = Lattice::new(&edges);

        let mut pos_bytes_to_chars = vec![0; s.as_raw_text().len() + 1];
        let mut pos_chars_to_bytes = vec![0];
        let mut pos = 0;
        for (i, c) in s.as_raw_text().chars().enumerate() {
            pos += c.len_utf8();
            pos_bytes_to_chars[pos] = i + 1;
            pos_chars_to_bytes.push(pos);
        }

        for m in pma.find_overlapping_iter(s.as_raw_text()) {
            let start = pos_bytes_to_chars[m.start()];
            let end = pos_bytes_to_chars[m.end()];
            let trunk_label = *trunk_labels.get(&(start, end)).unwrap_or(&0);
            for &label in &labels[m.value()] {
                if label == trunk_label {
                    continue;
                }
                lattice.add_branch(start, Edge::new(end, NonZeroUsize::new(label)));
            }
            for feature in feature_extractor.features_mut(
                s.as_raw_text(),
                s.char_types(),
                start,
                end,
                &pos_chars_to_bytes,
            ) {
                lattice.add_feature(start, end, feature);
            }
        }

        lattices.push(lattice);
    }

    let model = trainer.train(&lattices);

    // Removes unnecessary feature n-grams
    let mut removable_ids = vec![true; feature_extractor.n_ids];
    for hm in &model.unigram_fids {
        for &feature_id in hm.keys() {
            removable_ids[feature_id] = false;
        }
    }
    let mut new_self_feature_ids = HashMap::new();
    for (s, id) in feature_extractor.self_feature_ids {
        if removable_ids[id] {
            continue;
        }
        new_self_feature_ids.insert(s, id);
    }
    feature_extractor.self_feature_ids = new_self_feature_ids;
    let mut new_char_feature_ids = HashMap::new();
    for (s, hm) in feature_extractor.char_feature_ids {
        let mut new_hm = HashMap::new();
        for (k, id) in hm {
            if removable_ids[id] {
                continue;
            }
            new_hm.insert(k, id);
        }
        if new_hm.is_empty() {
            continue;
        }
        new_char_feature_ids.insert(s, new_hm);
    }
    feature_extractor.char_feature_ids = new_char_feature_ids;
    let mut new_type_feature_ids = HashMap::new();
    for (s, hm) in feature_extractor.type_feature_ids {
        let mut new_hm = HashMap::new();
        for (k, id) in hm {
            if removable_ids[id] {
                continue;
            }
            new_hm.insert(k, id);
        }
        if new_hm.is_empty() {
            continue;
        }
        new_type_feature_ids.insert(s, new_hm);
    }
    feature_extractor.type_feature_ids = new_type_feature_ids;

    MAModel {
        model,
        feature_extractor,
        surfaces,
        labels,
        tags: label_tag_map,
    }
}
