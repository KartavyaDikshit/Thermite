import os
import glob

def apply_wrapper():
    for f in glob.glob("thermite/*.py"):
        if f.endswith("__init__.py"): continue
        with open(f, "r") as file:
            content = file.read()
        
        if "def _catch_panic" not in content:
            new_content = """import numpy as np
import warnings
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        # basic input validation for all estimators
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.size == 0:
                raise ValueError("Empty input")
            if isinstance(arg, (list, tuple)) and len(arg) == 0:
                raise ValueError("Empty input")
        
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            err_str = str(e).lower()
            if "panic" in err_str or "empty" in err_str or "bounds" in err_str or "singular" in err_str or "invalid" in err_str:
                raise ValueError(str(e))
            raise
    return wrapper

""" + content.replace("import numpy as np\nfrom . import _core\n", "").replace("import numpy as np\nimport warnings\nfrom . import _core\n", "").replace("import numpy as np\nimport itertools\nfrom concurrent.futures import ThreadPoolExecutor\nfrom . import _core\n", "import itertools\nfrom concurrent.futures import ThreadPoolExecutor\n")
            
            # Now wrap all fit/predict methods
            new_content = new_content.replace("    def fit(self, ", "    @_catch_panic\n    def fit(self, ")
            new_content = new_content.replace("    def predict(self, ", "    @_catch_panic\n    def predict(self, ")
            new_content = new_content.replace("    def predict_proba(self, ", "    @_catch_panic\n    def predict_proba(self, ")
            new_content = new_content.replace("    def transform(self, ", "    @_catch_panic\n    def transform(self, ")
            new_content = new_content.replace("    def fit_predict(self, ", "    @_catch_panic\n    def fit_predict(self, ")
            
            with open(f, "w") as file:
                file.write(new_content)

apply_wrapper()
