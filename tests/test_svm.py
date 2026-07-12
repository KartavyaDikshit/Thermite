import numpy as np
import pytest
from sklearn.datasets import make_blobs
from thermite.svm import SVC

def test_svm_rbf_binary():
    # Generate simple binary dataset
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
    
    # Train SVC with RBF kernel and probability estimation
    clf = SVC(kernel='rbf', C=1.0, probability=True, tol=1e-3, max_iter=1000)
    clf.fit(X, y)
    
    # Check classes_ attribute
    assert list(clf.classes_) == [0.0, 1.0]
    
    # Check predictions
    preds = clf.predict(X)
    assert preds.shape == (100,)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.90, f"RBF binary accuracy was {accuracy}, expected > 0.90"
    
    # Check probability predictions
    probas = clf.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.allclose(np.sum(probas, axis=1), 1.0)
    assert np.all(probas >= 0.0)
    
    # Verify probability is higher for the predicted class
    pred_indices = np.argmax(probas, axis=1)
    # Map predictions back to class labels (0.0 or 1.0)
    pred_labels = np.array([clf.classes_[idx] for idx in pred_indices])
    assert np.all(pred_labels == preds)

def test_svm_poly_binary():
    # Generate simple binary dataset
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
    
    # Train SVC with Poly kernel
    clf = SVC(kernel='poly', degree=2, C=1.0, probability=False, tol=1e-3, max_iter=1000)
    clf.fit(X, y)
    
    # Check predictions
    preds = clf.predict(X)
    assert preds.shape == (100,)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.90, f"Poly binary accuracy was {accuracy}, expected > 0.90"

def test_svm_multiclass():
    # Generate simple 3-class dataset
    X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
    
    # Train SVC with RBF kernel and probability estimation
    clf = SVC(kernel='rbf', C=1.0, probability=True, tol=1e-3, max_iter=1000)
    clf.fit(X, y)
    
    # Check classes_ attribute
    assert len(clf.classes_) == 3
    
    # Check predictions
    preds = clf.predict(X)
    assert preds.shape == (150,)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.90, f"RBF multiclass accuracy was {accuracy}, expected > 0.90"
    
    # Check probability predictions
    probas = clf.predict_proba(X)
    assert probas.shape == (150, 3)
    assert np.allclose(np.sum(probas, axis=1), 1.0)
    assert np.all(probas >= 0.0)
    
    # Verify probability is higher for the predicted class
    pred_indices = np.argmax(probas, axis=1)
    pred_labels = np.array([clf.classes_[idx] for idx in pred_indices])
    assert np.all(pred_labels == preds)

if __name__ == "__main__":
    test_svm_rbf_binary()
    test_svm_poly_binary()
    test_svm_multiclass()
    print("All python SVM tests passed successfully!")
