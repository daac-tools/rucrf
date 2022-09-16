pub trait WeightVector {
    fn get_weight(&self, index: usize) -> f64;
    fn set_weight(&mut self, index: usize, value: f64);
}

impl WeightVector for Vec<f64> {
    #[inline(always)]
    fn get_weight(&self, index: usize) -> f64 {
        self[index]
    }

    #[inline(always)]
    fn set_weight(&mut self, index: usize, value: f64) {
        self[index] = value;
    }
}

pub struct RegularizedWeightVector<F> {
    step: usize,
    last_update: Vec<usize>,
    weights: Vec<f64>,
    f: F,
}

impl<F> RegularizedWeightVector<F>
where
    F: Fn(f64, usize, usize) -> f64,
{
    #[inline(always)]
    pub fn new(n: usize, f: F) -> Self {
        Self {
            step: 0,
            last_update: vec![0; n],
            weights: vec![0.0; n],
            f,
        }
    }

    #[inline(always)]
    pub fn increment_step(&mut self) {
        self.step += 1;
    }
}

impl<F> WeightVector for RegularizedWeightVector<F>
where
    F: Fn(f64, usize, usize) -> f64,
{
    #[inline(always)]
    fn get_weight(&self, index: usize) -> f64 {
        (self.f)(self.weights[index], self.step, self.last_update[index])
    }

    #[inline(always)]
    fn set_weight(&mut self, index: usize, value: f64) {
        self.weights[index] = value;
        self.last_update[index] = self.step;
    }
}

pub trait GradientVector {
    fn add(&mut self, index: usize, value: f64);
}

impl GradientVector for Vec<f64> {
    #[inline(always)]
    fn add(&mut self, index: usize, value: f64) {
        self[index] += value;
    }
}

#[derive(Default)]
pub struct SparseGrdientVector {
    gradients: Vec<(usize, f64)>,
}

impl SparseGrdientVector {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn apply_gradients<W>(&mut self, weights: &mut W, eta: f64)
    where
        W: WeightVector,
    {
        for &(index, value) in &self.gradients {
            weights.set_weight(index, weights.get_weight(index) - eta * value);
        }
        self.gradients.clear();
    }

    #[inline(always)]
    pub fn merge_gradients<G>(&mut self, other: &mut G)
    where
        G: GradientVector,
    {
        for &(index, value) in &self.gradients {
            other.add(index, value);
        }
        self.gradients.clear();
    }
}

impl GradientVector for SparseGrdientVector {
    #[inline(always)]
    fn add(&mut self, index: usize, value: f64) {
        self.gradients.push((index, value));
    }
}

impl<F> From<RegularizedWeightVector<F>> for Vec<f64> {
    fn from(v: RegularizedWeightVector<F>) -> Self {
        v.weights
    }
}
