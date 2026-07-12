pub mod preprocessing;
pub mod model_selection;

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

