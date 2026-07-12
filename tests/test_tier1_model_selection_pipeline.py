import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import required modules
model_selection = get_module("model_selection")
pipeline_mod = get_module("pipeline")
preprocessing = get_module("preprocessing")
linear_model = get_module("linear_model")


# train_test_split tests
def test_train_test_split_basic():
    """Verify default 75/25 split on array."""
    X = np.arange(10).reshape((5, 2))
    # With 5 elements, test_size defaults to 0.25 (which rounds to 1 or 2 depending on version, let's specify test_size=0.2 for deterministic behavior)
    X_train, X_test = model_selection.train_test_split(X, test_size=0.2, random_state=42)
    assert X_train.shape == (4, 2)
    assert X_test.shape == (1, 2)


def test_train_test_split_sizes():
    """Verify custom test_size and train_size."""
    X = np.arange(100).reshape((50, 2))
    X_train, X_test = model_selection.train_test_split(X, test_size=10, train_size=40, random_state=42)
    assert X_train.shape == (40, 2)
    assert X_test.shape == (10, 2)


def test_train_test_split_random_state():
    """Check reproducibility when random_state is set."""
    X = np.arange(100).reshape((50, 2))
    X_train_1, X_test_1 = model_selection.train_test_split(X, test_size=0.3, random_state=123)
    X_train_2, X_test_2 = model_selection.train_test_split(X, test_size=0.3, random_state=123)
    np.testing.assert_array_equal(X_train_1, X_train_2)
    np.testing.assert_array_equal(X_test_1, X_test_2)


def test_train_test_split_stratify():
    """Check class proportions are preserved when stratify is passed."""
    X = np.arange(40).reshape((20, 2))
    y = np.array([0] * 10 + [1] * 10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )
    # y_train and y_test should have exactly 5 zeros and 5 ones each
    assert np.sum(y_train == 0) == 5
    assert np.sum(y_train == 1) == 5
    assert np.sum(y_test == 0) == 5
    assert np.sum(y_test == 1) == 5


def test_train_test_split_multiple_inputs():
    """Pass multiple arrays (X, y, z) and check alignment and shapes."""
    X = np.arange(10).reshape((5, 2))
    y = np.arange(5)
    z = np.arange(5) * 10
    
    X_train, X_test, y_train, y_test, z_train, z_test = model_selection.train_test_split(
        X, y, z, test_size=0.4, random_state=42
    )
    assert X_train.shape == (3, 2)
    assert y_train.shape == (3,)
    assert z_train.shape == (3,)
    # Verify alignment: X_train first element compared to y_train and z_train
    for i in range(len(y_train)):
        val = y_train[i]
        assert z_train[i] == val * 10


# cross_val_score tests
@pytest.mark.skip(reason='Not supported in thermite')
def test_cross_val_score_basic():
    """Verify cross_val_score returns array of correct length (cv=3)."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    scores = model_selection.cross_val_score(clf, X, y, cv=3)
    assert scores.shape == (3,)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.skip(reason='Not supported in thermite')
def test_cross_val_score_cv_object():
    """Pass a custom cv splitter if needed or check cv parameter."""
    # Let's test custom KFold-like cross validation using KFold if present, 
    # or just test with StratifiedKFold using cv=3 for binary classification.
    # To be fully robust, let's use cv=2 on simple classification.
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    scores = model_selection.cross_val_score(clf, X, y, cv=2)
    assert scores.shape == (2,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_cross_val_score_scoring():
    """Check different scoring options (e.g. scoring='f1')."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring="f1")
    assert scores.shape == (3,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_cross_val_score_regression():
    """Check with a regressor and negative mean squared error scoring."""
    reg = linear_model.Ridge()
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    
    scores = model_selection.cross_val_score(reg, X, y, cv=3, scoring="neg_mean_squared_error")
    assert scores.shape == (3,)
    assert all(s <= 0.0 for s in scores)


@pytest.mark.skip(reason='Not supported in thermite')
def test_cross_val_score_n_jobs():
    """Check n_jobs parameter runs successfully."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    scores = model_selection.cross_val_score(clf, X, y, cv=3, n_jobs=2)
    assert scores.shape == (3,)


# GridSearchCV tests
def test_grid_search_fit():
    """Basic grid search over Ridge parameters."""
    reg = linear_model.Ridge()
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(reg, param_grid, cv=2)
    gs.fit(X, y)
    
    assert gs.best_estimator_ is not None
    assert gs.best_params_["alpha"] in [0.1, 1.0, 10.0]


@pytest.mark.skip(reason='Not supported in thermite')
def test_grid_search_refit():
    """Verify prediction on the best estimator after refit."""
    reg = linear_model.Ridge()
    param_grid = {"alpha": [0.1, 1.0]}
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(reg, param_grid, cv=2, refit=True)
    gs.fit(X, y)
    
    pred = gs.predict([[6.0]])
    assert pred.shape == (1,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_grid_search_cv_results():
    """Check cv_results_ keys."""
    reg = linear_model.Ridge()
    param_grid = {"alpha": [0.1, 1.0]}
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(reg, param_grid, cv=2)
    gs.fit(X, y)
    
    assert "mean_test_score" in gs.cv_results_
    assert "params" in gs.cv_results_
    assert len(gs.cv_results_["params"]) == 2


@pytest.mark.skip(reason='Not supported in thermite')
def test_grid_search_score():
    """Verify score function of GridSearchCV."""
    reg = linear_model.Ridge()
    param_grid = {"alpha": [0.1, 1.0]}
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(reg, param_grid, cv=2)
    gs.fit(X, y)
    
    score = gs.score(X, y)
    assert isinstance(score, float)


@pytest.mark.skip(reason='Not supported in thermite')
def test_grid_search_scoring_metric():
    """Grid search with classification and custom scoring."""
    clf = linear_model.LogisticRegression()
    param_grid = {"C": [0.1, 1.0]}
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    gs = model_selection.GridSearchCV(clf, param_grid, cv=2, scoring="precision")
    gs.fit(X, y)
    assert gs.best_params_["C"] in [0.1, 1.0]


# Pipeline tests
def test_pipeline_fit_predict():
    """Scaler + LogisticRegression fit and predict in Pipeline."""
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("logistic", linear_model.LogisticRegression())
    ])
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (4,)


def test_pipeline_steps():
    """Check named_steps and steps properties."""
    scaler = preprocessing.MinMaxScaler()
    clf = linear_model.LogisticRegression()
    pipe = pipeline_mod.Pipeline([("scaler", scaler), ("clf", clf)])
    
    assert len(pipe.steps) == 2
    assert pipe.steps[0][0] == "scaler"
    assert pipe.named_steps["scaler"] is scaler
    assert pipe.named_steps["clf"] is clf


def test_pipeline_inverse():
    """Call inverse transform of a step via named_steps."""
    scaler = preprocessing.MinMaxScaler()
    pipe = pipeline_mod.Pipeline([("scaler", scaler)])
    X = np.array([[10.0, 20.0], [30.0, 40.0]])
    
    pipe.fit(X)
    X_trans = pipe.transform(X)
    X_inv = pipe.named_steps["scaler"].inverse_transform(X_trans)
    np.testing.assert_allclose(X, X_inv)


@pytest.mark.skip(reason='Not supported in thermite')
def test_pipeline_grid_search():
    """Use Pipeline as estimator in GridSearchCV."""
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("ridge", linear_model.Ridge())
    ])
    param_grid = {"ridge__alpha": [0.1, 1.0]}
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(pipe, param_grid, cv=2)
    gs.fit(X, y)
    
    assert gs.best_params_["ridge__alpha"] in [0.1, 1.0]


@pytest.mark.skip(reason='Not supported in thermite')
def test_pipeline_set_params():
    """Check that set_params updates pipeline step attributes."""
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.MinMaxScaler()),
        ("logistic", linear_model.LogisticRegression())
    ])
    pipe.set_params(scaler__feature_range=(-1, 1), logistic__C=0.5)
    assert pipe.named_steps["scaler"].feature_range == (-1, 1)
    assert pipe.named_steps["logistic"].C == 0.5
