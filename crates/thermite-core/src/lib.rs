pub mod preprocessing;
pub mod linear_model;
pub mod tree;
pub mod cluster;
pub mod metrics;
pub mod model_selection;
pub mod decomposition;
pub mod neighbors;
pub mod ensemble;
pub mod sparse;
pub mod naive_bayes;

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

