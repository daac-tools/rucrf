use core::num::NonZeroUsize;

use alloc::vec::Vec;

use hashbrown::HashMap;

/// Represents an edge in a lattice
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Edge {
    target: usize,
    pub(crate) label: usize,
}

impl Edge {
    /// Creates a new edge
    ///
    /// # Arguments
    ///
    /// * `target` - Index of the node to which the edge points.
    /// * `label` - Label this edge holds.
    #[inline(always)]
    pub fn new(target: usize, label: Option<NonZeroUsize>) -> Self {
        Self {
            target,
            label: label.map_or(0, |x| x.get()),
        }
    }

    /// Gets the target value
    #[inline(always)]
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Gets the label value
    #[inline(always)]
    pub const fn label(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.label)
    }
}

#[derive(Clone, Default, Debug)]
pub struct Node {
    edges: Vec<Edge>,
}

impl Node {
    #[inline(always)]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Represents a feature
#[derive(Debug)]
pub struct Feature {
    /// Feature ID
    pub feature_id: usize,

    /// Feature value
    pub value: f64,
}

impl Feature {
    /// Creates a new feature with its ID and value
    #[inline(always)]
    pub fn new(feature_id: usize, value: f64) -> Self {
        Self { feature_id, value }
    }
}

/// Represents a lattice
#[derive(Debug)]
pub struct Lattice {
    nodes: Vec<Node>,
    features: HashMap<(usize, usize), Vec<Feature>>,
}

impl Lattice {
    /// Creates a new lattice
    ///
    /// During training, the path specified here is treated as a positive example.
    ///
    /// # Arguments
    ///
    /// * `edges` - List of edges representing a single path.
    ///
    /// # Panics
    ///
    /// Target of each edge must point backward.
    pub fn new(edges: &[Edge]) -> Self {
        assert!(!edges.is_empty());
        let nodes_len = edges.last().unwrap().target() + 1;
        let mut nodes = vec![Node::default(); nodes_len];
        let mut pos = 0;
        for &edge in edges {
            nodes[pos].edges.push(edge);
            assert!(edge.target() > pos);
            pos = edge.target();
        }
        Self {
            nodes,
            features: HashMap::new(),
        }
    }

    /// Adds a new branch
    ///
    /// During training, the path specified here is treated as a negative example.
    ///
    /// # Arguments
    ///
    /// * `start` - Index of the starting point of the edge.
    /// * `edge` - Edge to add.
    ///
    /// # Panics
    ///
    /// Target of the each must point backward.
    #[inline(always)]
    pub fn add_branch(&mut self, start: usize, edge: Edge) {
        assert!(start < edge.target());
        self.nodes[start].edges.push(edge);
    }

    /// Adds a feature for the set of edges connecting the specified start and end points.
    ///
    /// # Arguments
    ///
    /// * `start` - Start point.
    /// * `end` - End point.
    /// * `feature` - Feature to add.
    ///
    /// # Panics
    ///
    /// `end` must be greater than `start`.
    #[inline(always)]
    pub fn add_feature(&mut self, start: usize, end: usize, feature: Feature) {
        assert!(start < end);
        self.features
            .entry((start, end))
            .or_insert_with(|| vec![])
            .push(feature);
    }

    #[inline(always)]
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    #[inline(always)]
    pub(crate) fn features(&self) -> &HashMap<(usize, usize), Vec<Feature>> {
        &self.features
    }
}
