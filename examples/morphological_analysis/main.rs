mod feature_extractor;
mod trainer;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use bincode::{Decode, Encode};
use clap::Parser;
use daachorse::DoubleArrayAhoCorasick;
use hashbrown::HashMap;
use rucrf::{Edge, Lattice, Model};
use vaporetto::{CharacterType, Sentence};

use feature_extractor::FeatureExtractor;

#[derive(Decode, Encode)]
pub struct MAModel {
    model: Model,
    feature_extractor: FeatureExtractor,
    surfaces: Vec<String>,
    labels: Vec<Vec<usize>>,
    tags: Vec<String>,
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the model file
    #[clap(long, value_parser)]
    model: PathBuf,

    /// Path to the corpus file
    #[clap(long, value_parser)]
    corpus: Option<PathBuf>,

    /// Path to the dictionary file
    #[clap(long, value_parser)]
    dict: Option<PathBuf>,

    /// Number of threads
    #[clap(long, value_parser)]
    n_threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Some(corpus) = args.corpus {
        let f = File::open(corpus)?;
        let f = BufReader::new(f);
        let mut corpus_sents = vec![];
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {}\r", i);
                io::stderr().flush()?;
            }
            corpus_sents.push(Sentence::from_tokenized(&line?)?);
        }
        eprintln!("# of sentences: {}", corpus_sents.len());
        let mut dict_sents = vec![];
        if let Some(dict) = args.dict {
            let f = File::open(dict)?;
            let f = BufReader::new(f);
            for (i, line) in f.lines().enumerate() {
                if i % 100000 == 0 {
                    eprint!("# of words: {}\r", i);
                    io::stderr().flush()?;
                }
                dict_sents.push(Sentence::from_tokenized(&line?)?);
            }
            eprintln!("# of words: {}", dict_sents.len());
        }
        let model = trainer::train(&corpus_sents, &dict_sents, args.n_threads);

        let mut f = zstd::Encoder::new(File::create(args.model)?, 19)?;
        bincode::encode_into_std_write(model, &mut f, bincode::config::standard()).unwrap();
        f.finish()?;
    } else {
        let mut f = zstd::Decoder::new(File::open(args.model)?)?;
        let model: MAModel =
            bincode::decode_from_std_read(&mut f, bincode::config::standard()).unwrap();

        let mut surface_ids = HashMap::new();
        for (i, surf) in model.surfaces.iter().enumerate() {
            surface_ids.insert(surf.to_string(), i);
        }
        let mut tag_ids = HashMap::new();
        for (i, tag) in model.tags.iter().enumerate() {
            tag_ids.insert(tag.to_string(), i + 1);
        }

        let pma = DoubleArrayAhoCorasick::<usize>::new(&model.surfaces).unwrap();

        for line in io::stdin().lock().lines() {
            let line = line?;
            let mut pos_bytes_to_chars = vec![0; line.len() + 1];
            let mut pos_chars_to_bytes = vec![0];
            let mut pos = 0;
            let mut types = vec![];
            for (i, c) in line.chars().enumerate() {
                pos += c.len_utf8();
                pos_bytes_to_chars[pos] = i + 1;
                pos_chars_to_bytes.push(pos);
                types.push(CharacterType::get_type(c) as u8);
            }
            let mut lattice = Lattice::new(&[Edge::new(pos_chars_to_bytes.len() - 1, None)]);

            for m in pma.find_overlapping_iter(&line) {
                let start = pos_bytes_to_chars[m.start()];
                let end = pos_bytes_to_chars[m.end()];
                for &label in &model.labels[m.value()] {
                    lattice.add_branch(start, Edge::new(end, NonZeroUsize::new(label)));
                }
                for feature in
                    model
                        .feature_extractor
                        .features(&line, &types, start, end, &pos_chars_to_bytes)
                {
                    lattice.add_feature(start, end, feature);
                }
            }

            let mut pos = 0;
            let path = model.model.search_best_path(&lattice);
            for edge in path {
                println!(
                    "{}/{}",
                    &line[pos_chars_to_bytes[pos]..pos_chars_to_bytes[edge.target()]],
                    &model.tags[edge.label().unwrap().get() - 1]
                );
                pos = edge.target();
            }
        }
    }
    Ok(())
}
