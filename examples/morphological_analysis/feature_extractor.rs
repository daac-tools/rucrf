use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    Decode, Encode,
};
use hashbrown::HashMap;
use rucrf::Feature;

#[derive(Debug)]
pub struct FeatureExtractor {
    ngram_size: usize,
    pub self_feature_ids: HashMap<String, usize>,
    pub char_feature_ids: HashMap<String, HashMap<(usize, bool, bool), usize>>,
    pub type_feature_ids: HashMap<Vec<u8>, HashMap<(usize, bool, bool), usize>>,
    pub n_ids: usize,
}

impl FeatureExtractor {
    pub fn new(ngram_size: usize) -> Self {
        Self {
            ngram_size,
            self_feature_ids: HashMap::new(),
            char_feature_ids: HashMap::new(),
            type_feature_ids: HashMap::new(),
            n_ids: 0,
        }
    }

    pub fn features_mut(
        &mut self,
        text: &str,
        types: &[u8],
        start: usize,
        end: usize,
        pos_chars_to_bytes: &[usize],
    ) -> Vec<Feature> {
        let mut result = vec![];
        let ngram = &text[pos_chars_to_bytes[start]..pos_chars_to_bytes[end]];
        let feature_id = *self
            .self_feature_ids
            .raw_entry_mut()
            .from_key(ngram)
            .or_insert_with(|| (ngram.to_string(), self.n_ids))
            .1;
        if feature_id == self.n_ids {
            self.n_ids += 1;
        }
        result.push(Feature::new(feature_id, 2.0));
        for n in 1..=self.ngram_size {
            for i in 0..=n {
                if start + i < n - 1 || end + i > types.len() + 1 {
                    continue;
                }
                let (range_start, bos) = if start + i == n - 1 {
                    (0, true)
                } else {
                    (start + i - n, false)
                };
                let (range_end, eos) = if end + i == types.len() + 1 {
                    (types.len(), true)
                } else {
                    (end + i, false)
                };
                let ngram = &text[pos_chars_to_bytes[range_start]..pos_chars_to_bytes[range_end]];
                let feature_id = *self
                    .char_feature_ids
                    .raw_entry_mut()
                    .from_key(ngram)
                    .or_insert_with(|| (ngram.to_string(), HashMap::new()))
                    .1
                    .entry((i, bos, eos))
                    .or_insert(self.n_ids);
                if feature_id == self.n_ids {
                    self.n_ids += 1;
                }
                result.push(Feature::new(feature_id, 1.0));
                let ngram = &types[range_start..range_end];
                let feature_id = *self
                    .type_feature_ids
                    .raw_entry_mut()
                    .from_key(ngram)
                    .or_insert_with(|| (ngram.to_vec(), HashMap::new()))
                    .1
                    .entry((i, bos, eos))
                    .or_insert(self.n_ids);
                if feature_id == self.n_ids {
                    self.n_ids += 1;
                }
                result.push(Feature::new(feature_id, 1.0));
            }
        }
        result
    }

    pub fn features(
        &self,
        text: &str,
        types: &[u8],
        start: usize,
        end: usize,
        pos_chars_to_bytes: &[usize],
    ) -> Vec<Feature> {
        let mut result = vec![];
        let ngram = &text[pos_chars_to_bytes[start]..pos_chars_to_bytes[end]];
        if let Some(&feature_id) = self.self_feature_ids.get(ngram) {
            result.push(Feature::new(feature_id, 2.0));
        }
        for n in 1..=self.ngram_size {
            for i in 0..=n {
                if start + i < n - 1 || end + i > types.len() + 1 {
                    continue;
                }
                let (range_start, bos) = if start + i == n - 1 {
                    (0, true)
                } else {
                    (start + i - n, false)
                };
                let (range_end, eos) = if end + i == types.len() + 1 {
                    (types.len(), true)
                } else {
                    (end + i, false)
                };
                let ngram = &text[pos_chars_to_bytes[range_start]..pos_chars_to_bytes[range_end]];
                if let Some(&feature_id) = self
                    .char_feature_ids
                    .get(ngram)
                    .and_then(|hm| hm.get(&(i, bos, eos)))
                {
                    result.push(Feature::new(feature_id, 1.0));
                }
                let ngram = &types[range_start..range_end];
                if let Some(&feature_id) = self
                    .type_feature_ids
                    .get(ngram)
                    .and_then(|hm| hm.get(&(i, bos, eos)))
                {
                    result.push(Feature::new(feature_id, 1.0));
                }
            }
        }
        result
    }
}

impl Decode for FeatureExtractor {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let ngram_size = Decode::decode(decoder)?;
        let self_feature_ids: Vec<(String, usize)> = Decode::decode(decoder)?;
        let char_feature_ids: Vec<(String, Vec<((usize, bool, bool), usize)>)> =
            Decode::decode(decoder)?;
        let type_feature_ids: Vec<(Vec<u8>, Vec<((usize, bool, bool), usize)>)> =
            Decode::decode(decoder)?;
        let n_ids = Decode::decode(decoder)?;
        Ok(Self {
            ngram_size,
            self_feature_ids: self_feature_ids.into_iter().collect(),
            char_feature_ids: char_feature_ids
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect(),
            type_feature_ids: type_feature_ids
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect(),
            n_ids,
        })
    }
}

impl Encode for FeatureExtractor {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.ngram_size, encoder)?;
        let self_feature_ids: Vec<(String, usize)> =
            self.self_feature_ids.clone().into_iter().collect();
        let char_feature_ids: Vec<(String, Vec<((usize, bool, bool), usize)>)> = self
            .char_feature_ids
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().into_iter().collect()))
            .collect();
        let type_feature_ids: Vec<(Vec<u8>, Vec<((usize, bool, bool), usize)>)> = self
            .type_feature_ids
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().into_iter().collect()))
            .collect();
        Encode::encode(&self_feature_ids, encoder)?;
        Encode::encode(&char_feature_ids, encoder)?;
        Encode::encode(&type_feature_ids, encoder)?;
        Encode::encode(&self.n_ids, encoder)?;
        Ok(())
    }
}
