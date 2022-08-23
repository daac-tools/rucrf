//! # rucrf
//!
//! Conditional Random Fields (CRFs) implemented in pure Rust
#![cfg_attr(
    all(feature = "std", feature = "tag-prediction"),
    doc = "
## Examples

```rust
use std::num::NonZeroUsize;

use rucrf::{Edge, Feature, Lattice, Trainer};

// Train:
// 京(kyo) 都(to)
// 東(to) 京(kyo)
// 京(kei) 浜(hin)
// 京(kyo) の(no) 都(miyako)
//
// Test:
// 水(mizu) の(no) 都(miyako)
//
// Features:
// 京: 0, 都: 1, 東: 2, 浜: 3, の: 4, 水: 5
//
// Labels:
// kyo: 1, to: 2, kei: 3, hin: 4, no: 5, mikako: 6, mizu: 7

let mut lattices = vec![];

// 京都 (kyo to)
let mut lattice = Lattice::new(&[
    Edge::new(1, NonZeroUsize::new(1)),
    Edge::new(2, NonZeroUsize::new(2)),
]);

// add other edges
lattice.add_branch(0, Edge::new(1, NonZeroUsize::new(3))); // kei
lattice.add_branch(1, Edge::new(2, NonZeroUsize::new(6))); // miyako

// add features
lattice.add_feature(0, 1, Feature::new(0, 1.0));
lattice.add_feature(1, 2, Feature::new(1, 1.0));
lattices.push(lattice);

// 東京 (to kyo)
let mut lattice = Lattice::new(&[
    Edge::new(1, NonZeroUsize::new(2)),
    Edge::new(2, NonZeroUsize::new(1)),
]);

// add other edges
lattice.add_branch(1, Edge::new(2, NonZeroUsize::new(3))); // kei

// add features
lattice.add_feature(0, 1, Feature::new(2, 1.0));
lattice.add_feature(1, 2, Feature::new(0, 1.0));
lattices.push(lattice);

// 京浜 (kei hin)
let mut lattice = Lattice::new(&[
    Edge::new(1, NonZeroUsize::new(3)),
    Edge::new(2, NonZeroUsize::new(4)),
]);

// add other edges
lattice.add_branch(0, Edge::new(1, NonZeroUsize::new(1))); // kyo

// add features
lattice.add_feature(0, 1, Feature::new(0, 1.0));
lattice.add_feature(1, 2, Feature::new(3, 1.0));
lattices.push(lattice);

// 京の都 (kyo no miyako)
let mut lattice = Lattice::new(&[
    Edge::new(1, NonZeroUsize::new(1)),
    Edge::new(2, NonZeroUsize::new(5)),
    Edge::new(3, NonZeroUsize::new(6)),
]);

// add other edges
lattice.add_branch(0, Edge::new(1, NonZeroUsize::new(3))); // kei
lattice.add_branch(2, Edge::new(3, NonZeroUsize::new(2))); // to

// add features
lattice.add_feature(0, 1, Feature::new(0, 1.0));
lattice.add_feature(1, 2, Feature::new(4, 1.0));
lattice.add_feature(2, 3, Feature::new(1, 1.0));
lattices.push(lattice);

// Generates a model
let trainer = Trainer::new();
let model = trainer.train(&lattices);

// 水の都 (mizu no miyako)
let mut lattice = Lattice::new(&[Edge::new(3, None)]);

// add edges
lattice.add_branch(0, Edge::new(1, NonZeroUsize::new(7))); // mizu
lattice.add_branch(1, Edge::new(2, NonZeroUsize::new(5))); // no
lattice.add_branch(2, Edge::new(3, NonZeroUsize::new(2))); // to
lattice.add_branch(2, Edge::new(3, NonZeroUsize::new(6))); // miyako

// add features
lattice.add_feature(0, 1, Feature::new(5, 1.0));
lattice.add_feature(1, 2, Feature::new(4, 1.0));
lattice.add_feature(2, 3, Feature::new(1, 1.0));

let path = model.search_best_path(&lattice);

assert_eq!(vec![
    Edge::new(1, NonZeroUsize::new(7)),
    Edge::new(2, NonZeroUsize::new(5)),
    Edge::new(3, NonZeroUsize::new(6)),
], path);
```
"
)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "alloc"))]
compile_error!("`alloc` feature is currently required to build this crate");

#[macro_use]
extern crate alloc;

mod lattice;
mod model;

#[cfg(feature = "train")]
mod forward_backward;
#[cfg(feature = "train")]
mod math;
#[cfg(feature = "train")]
mod trainer;

#[cfg(test)]
mod test_utils;

pub use lattice::{Edge, Feature, Lattice};
pub use model::Model;

#[cfg(feature = "train")]
pub use trainer::{Regularization, Trainer};

/// A specialized Result type.
pub type Result<T, E = &'static str> = core::result::Result<T, E>;
