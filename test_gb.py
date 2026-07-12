import numpy as np
from sklearn.datasets import make_regression, make_classification
from thermite.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as SkGBR, GradientBoostingClassifier as SkGBC

def test_gb():
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    print("Testing Thermite GradientBoostingRegressor...")
    th_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    th_gbr.fit(X_reg, y_reg)
    print("Thermite GBR R2:", th_gbr.score(X_reg, y_reg))

    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    print("Testing Thermite GradientBoostingClassifier...")
    th_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    th_gbc.fit(X_clf, y_clf)
    print("Thermite GBC Accuracy:", th_gbc.score(X_clf, y_clf))

if __name__ == "__main__":
    test_gb()
