#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod cluster;
pub mod decomposition;
pub mod automl;
pub mod ensemble;
pub mod linear_model;
pub mod metrics;
pub mod model_selection;
pub mod naive_bayes;
pub mod neighbors;
pub mod preprocessing;
pub mod sparse;
pub mod svm;
pub mod tree;

pub fn core_ping() -> &'static str {
    "core_pong"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_ping() {
        assert_eq!(core_ping(), "core_pong");
    }
}
