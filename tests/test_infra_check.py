from tests.conftest import get_module, USE_SKLEARN

def test_backend_switcher():
    """
    Verify that get_module returns modules from the appropriate package.
    """
    linear_model = get_module("linear_model")
    metrics = get_module("metrics")
    
    expected_prefix = "sklearn" if USE_SKLEARN else "thermite"
    
    assert linear_model.__name__.startswith(expected_prefix)
    assert metrics.__name__.startswith(expected_prefix)
    
    print(f"\nBackend verification successful. USE_SKLEARN={USE_SKLEARN}")
    print(f"linear_model resolved to: {linear_model.__name__}")
    print(f"metrics resolved to: {metrics.__name__}")
