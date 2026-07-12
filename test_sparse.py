import numpy as np
import scipy.sparse as sp
from thermite.linear_model import LogisticRegression

def test_sparse_logistic():
    # Create simple binary classification dataset
    X = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ], dtype=np.float64)
    y = np.array([0, 1, 1, 0, 0])
    
    # Dense fit
    clf_dense = LogisticRegression(C=1.0, max_iter=200)
    clf_dense.fit(X, y)
    
    # Sparse fit
    X_sparse = sp.csr_matrix(X)
    clf_sparse = LogisticRegression(C=1.0, max_iter=200)
    clf_sparse.fit(X_sparse, y)
    
    np.testing.assert_allclose(clf_dense.coef_, clf_sparse.coef_, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(clf_dense.intercept_, clf_sparse.intercept_, atol=1e-3, rtol=1e-3)
    
    pred_dense = clf_dense.predict(X)
    pred_sparse = clf_sparse.predict(X_sparse)
    
    np.testing.assert_array_equal(pred_dense, pred_sparse)
    
    proba_dense = clf_dense.predict_proba(X)
    proba_sparse = clf_sparse.predict_proba(X_sparse)
    
    np.testing.assert_allclose(proba_dense, proba_sparse, atol=1e-3, rtol=1e-3)
    
    print("All sparse logistic regression tests passed!")

def test_sparse_linearsvc():
    from thermite.linear_model import LinearSVC
    
    X = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ], dtype=np.float64)
    y = np.array([0, 1, 1, 0, 0])
    
    clf_dense = LinearSVC(C=1.0, max_iter=200)
    clf_dense.fit(X, y)
    
    X_sparse = sp.csr_matrix(X)
    clf_sparse = LinearSVC(C=1.0, max_iter=200)
    clf_sparse.fit(X_sparse, y)
    
    np.testing.assert_allclose(clf_dense.coef_, clf_sparse.coef_, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(clf_dense.intercept_, clf_sparse.intercept_, atol=1e-3, rtol=1e-3)
    
    pred_dense = clf_dense.predict(X)
    pred_sparse = clf_sparse.predict(X_sparse)
    
    np.testing.assert_array_equal(pred_dense, pred_sparse)
    
    print("All sparse LinearSVC tests passed!")

if __name__ == "__main__":
    test_sparse_logistic()
    test_sparse_linearsvc()
