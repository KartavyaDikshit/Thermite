pub mod preprocessing;
pub mod model_selection;
pub mod linear_model;
pub mod metrics;
pub mod tree;
pub mod cluster;
pub mod decomposition;
pub mod neighbors;

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

