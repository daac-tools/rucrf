# rucrf: Conditional Random Fields implemented in pure Rust

*rucrf* contains a trainer and an estimator for Conditional Random Fields (CRFs).
This library supports:
- [x] lattices with variable length edges,
- [x] L1 and L2 regularization, and
- [x] multi-threading during training.

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

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
