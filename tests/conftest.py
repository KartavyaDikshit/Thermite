import os
import importlib

# Determine if we should use scikit-learn or thermite
USE_SKLEARN = os.environ.get("USE_SKLEARN", "").lower() in ("1", "true")

def get_module(module_name: str):
    """
    Dynamically import a module from either 'sklearn' or 'thermite'
    depending on the USE_SKLEARN environment variable.

    For example:
        from tests.conftest import get_module
        linear_model = get_module('linear_model')
        
    This allows E2E tests to run against either sklearn (to verify test correctness)
    or thermite (to verify our implementation).
    """
    base_package = "sklearn" if USE_SKLEARN else "thermite"
    
    if not module_name:
        return importlib.import_module(base_package)
        
    full_module_name = f"{base_package}.{module_name}"
    try:
        return importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import {full_module_name} (backend: {base_package}). "
            f"Please ensure the package is correctly installed. Original error: {e}"
        )
