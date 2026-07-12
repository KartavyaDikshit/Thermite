import numpy as np
from . import _core

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets.
    
    Compatible with scikit-learn.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required as input")
        
    np_arrays = [np.asarray(arr) for arr in arrays]
    
    if stratify is not None:
        stratify = np.asarray(stratify)
        
    return _core.train_test_split(
        *np_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    )
