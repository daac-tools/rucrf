pub mod lbfgs;
pub mod momentum_sgd;
pub mod sgd;

pub enum LearningRateDecay {
    Inverse,
    Exponential(f64),
}
