import numpy as np
from sklearn.datasets import make_classification
from thermite.ensemble import RandomForestClassifier
from thermite.model_selection import GridSearchCV

def test_grid_search():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [None, 3]
    }
    
    print("Running GridSearchCV with n_jobs=1")
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=1)
    grid.fit(X, y)
    
    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)
    
    assert grid.best_params_ is not None
    assert grid.best_score_ > 0.8
    
    print("Running GridSearchCV with n_jobs=2")
    grid_par = GridSearchCV(rf, param_grid, cv=3, n_jobs=2)
    grid_par.fit(X, y)
    
    print("Best params (parallel):", grid_par.best_params_)
    print("Best score (parallel):", grid_par.best_score_)
    
    assert grid_par.best_params_ == grid.best_params_
    
if __name__ == "__main__":
    test_grid_search()
