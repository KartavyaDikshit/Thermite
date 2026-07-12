import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.svm import LinearSVC as SkLinearSVC
from thermite.linear_model import LogisticRegression, LinearSVC

def test_multiclass():
    # Create a 3-class dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=3, random_state=42)
    
    print("Testing SKLearn LogisticRegression Multiclass...")
    sk_lr = SkLogisticRegression(max_iter=100)
    sk_lr.fit(X, y)
    preds_sk_lr = sk_lr.predict(X)
    acc_sk_lr = np.mean(preds_sk_lr == y)
    print(f"SKLearn LogisticRegression Accuracy: {acc_sk_lr:.4f}")

    print("Testing Thermite LogisticRegression Multiclass...")
    lr = LogisticRegression(max_iter=100)
    lr.fit(X, y)
    preds_lr = lr.predict(X)
    acc_lr = np.mean(preds_lr == y)
    print(f"Thermite LogisticRegression Accuracy: {acc_lr:.4f}")
    assert acc_lr > 0.5, f"Accuracy too low: {acc_lr}"

    print("Testing SKLearn LinearSVC Multiclass...")
    sk_svc = SkLinearSVC(max_iter=1000)
    sk_svc.fit(X, y)
    preds_sk_svc = sk_svc.predict(X)
    acc_sk_svc = np.mean(preds_sk_svc == y)
    print(f"SKLearn LinearSVC Accuracy: {acc_sk_svc:.4f}")

    print("Testing Thermite LinearSVC Multiclass...")
    svc = LinearSVC(max_iter=1000)
    svc.fit(X, y)
    preds_svc = svc.predict(X)
    acc_svc = np.mean(preds_svc == y)
    print(f"Thermite LinearSVC Accuracy: {acc_svc:.4f}")

if __name__ == "__main__":
    test_multiclass()
