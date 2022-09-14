# rucrf: Conditional Random Fields implemented in pure Rust

*rucrf* contains a trainer and an estimator for Conditional Random Fields (CRFs).
This library supports:
- [x] lattices with variable length edges,
- [x] L1 and L2 regularization, and
- [x] multi-threading during training.

## Examples

```rust
use std::num::NonZeroU32;

use rucrf::{Edge, FeatureProvider, FeatureSet, Lattice, Model, Trainer};

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
// 京: 1, 都: 2, 東: 3, 浜: 4, の: 5, 水: 6
//
// Labels:
// 京kyo: 1, 都to: 2, 東to: 3,  京kei: 4, 浜hin: 5, のno: 6, 都mikako: 7, 水mizu: 8

let mut provider = FeatureProvider::new();
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(1).unwrap()],
    &[NonZeroU32::new(1)],
    &[NonZeroU32::new(1)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(2).unwrap()],
    &[NonZeroU32::new(2)],
    &[NonZeroU32::new(2)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(3).unwrap()],
    &[NonZeroU32::new(3)],
    &[NonZeroU32::new(3)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(1).unwrap()],
    &[NonZeroU32::new(4)],
    &[NonZeroU32::new(4)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(4).unwrap()],
    &[NonZeroU32::new(5)],
    &[NonZeroU32::new(5)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(5).unwrap()],
    &[NonZeroU32::new(6)],
    &[NonZeroU32::new(6)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(2).unwrap()],
    &[NonZeroU32::new(7)],
    &[NonZeroU32::new(7)],
));
provider.add_feature_set(FeatureSet::new(
    &[NonZeroU32::new(6).unwrap()],
    &[NonZeroU32::new(8)],
    &[NonZeroU32::new(8)],
));

let mut lattices = vec![];

// 京都 (kyo to)
let mut lattice = Lattice::new(2)?;
lattice.add_edge(0, Edge::new(1, NonZeroU32::new(1).unwrap()))?; // kyo
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(2).unwrap()))?; // to

lattice.add_edge(0, Edge::new(1, NonZeroU32::new(4).unwrap()))?; // kei
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(7).unwrap()))?; // miyako

lattices.push(lattice);

// 東京 (to kyo)
let mut lattice = Lattice::new(2)?;
lattice.add_edge(0, Edge::new(1, NonZeroU32::new(3).unwrap()))?; // to
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(1).unwrap()))?; // kyo

lattice.add_edge(1, Edge::new(2, NonZeroU32::new(4).unwrap()))?; // kei

lattices.push(lattice);

// 京浜 (kei hin)
let mut lattice = Lattice::new(2)?;
lattice.add_edge(0, Edge::new(1, NonZeroU32::new(4).unwrap()))?; // kei
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(5).unwrap()))?; // hin

lattice.add_edge(0, Edge::new(1, NonZeroU32::new(1).unwrap()))?; // kyo

lattices.push(lattice);

// 京の都 (kyo no miyako)
let mut lattice = Lattice::new(3)?;
lattice.add_edge(0, Edge::new(1, NonZeroU32::new(1).unwrap()))?; // kyo
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(6).unwrap()))?; // no
lattice.add_edge(2, Edge::new(3, NonZeroU32::new(7).unwrap()))?; // miyako

lattice.add_edge(0, Edge::new(1, NonZeroU32::new(4).unwrap()))?; // kei
lattice.add_edge(2, Edge::new(3, NonZeroU32::new(2).unwrap()))?; // to

lattices.push(lattice);

// Generates a model
let trainer = Trainer::new();
let model = trainer.train(&lattices, provider);

// 水の都 (mizu no miyako)
let mut lattice = Lattice::new(3)?;
lattice.add_edge(0, Edge::new(1, NonZeroU32::new(8).unwrap()))?; // mizu
lattice.add_edge(1, Edge::new(2, NonZeroU32::new(6).unwrap()))?; // no
lattice.add_edge(2, Edge::new(3, NonZeroU32::new(2).unwrap()))?; // to
lattice.add_edge(2, Edge::new(3, NonZeroU32::new(7).unwrap()))?; // miyako

let (path, _) = model.search_best_path(&lattice);

assert_eq!(vec![
    Edge::new(1, NonZeroU32::new(8).unwrap()),
    Edge::new(2, NonZeroU32::new(6).unwrap()),
    Edge::new(3, NonZeroU32::new(7).unwrap()),
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
