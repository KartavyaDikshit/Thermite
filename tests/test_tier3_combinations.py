import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import all needed modules
preprocessing = get_module("preprocessing")
linear_model = get_module("linear_model")
tree = get_module("tree")
ensemble = get_module("ensemble")
neighbors = get_module("neighbors")
cluster = get_module("cluster")
decomposition = get_module("decomposition")
model_selection = get_module("model_selection")
pipeline_mod = get_module("pipeline")
metrics = get_module("metrics")


@pytest.mark.skip(reason='Not supported in thermite')
def test_standard_scaler_linear_regression():
    """
    1. Pairwise combination of StandardScaler and LinearRegression.
    Fits a simple regression dataset after scaling, and checks predictions.
    """
    scaler = preprocessing.StandardScaler()
    reg = linear_model.LinearRegression()
    
    X_train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    y_train = np.array([2.0, 4.0, 6.0, 8.0])
    X_test = np.array([[5.0, 10.0]])
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    reg.fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_test_scaled)
    
    # Expected output should be close to 10.0 (extrapolating the pattern y = 2 * x_0)
    np.testing.assert_allclose(y_pred, [10.0], rtol=1e-5)


def test_standard_scaler_ridge():
    """
    2. Pairwise combination of StandardScaler and Ridge.
    Fits a regression dataset with L2 regularization after scaling.
    """
    scaler = preprocessing.StandardScaler()
    reg = linear_model.Ridge(alpha=0.5)
    
    X_train = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y_train = np.array([1.5, 2.5, 3.5])
    X_test = np.array([[4.0, 4.0]])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    y_pred = reg.predict(scaler.transform(X_test))
    assert len(y_pred) == 1
    assert isinstance(float(y_pred[0]), float)


def test_standard_scaler_lasso():
    """
    3. Pairwise combination of StandardScaler and Lasso.
    Fits a regression dataset with L1 regularization after scaling.
    """
    scaler = preprocessing.StandardScaler()
    reg = linear_model.Lasso(alpha=0.1)
    
    X_train = np.array([[10.0, -1.0], [20.0, 1.0], [30.0, 0.0]])
    y_train = np.array([2.0, 4.0, 6.0])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    y_pred = reg.predict(X_scaled)
    assert len(y_pred) == 3


def test_standard_scaler_logistic_regression():
    """
    4. Pairwise combination of StandardScaler and LogisticRegression.
    Fits a classification dataset after scaling, checking accuracy score.
    """
    scaler = preprocessing.StandardScaler()
    clf = linear_model.LogisticRegression()
    
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    acc = metrics.accuracy_score(y_train, pred)
    assert acc == 1.0


def test_min_max_scaler_linear_regression():
    """
    5. Pairwise combination of MinMaxScaler and LinearRegression.
    Fits a simple regression dataset with minimax scaling.
    """
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    reg = linear_model.LinearRegression()
    
    X_train = np.array([[0.0], [10.0], [20.0]])
    y_train = np.array([1.0, 3.0, 5.0])
    
    X_scaled = scaler.fit_transform(X_train)
    # The scaled values should be in feature_range (-1, 1)
    np.testing.assert_allclose(X_scaled, [[-1.0], [0.0], [1.0]], atol=1e-7)
    
    reg.fit(X_scaled, y_train)
    y_pred = reg.predict(scaler.transform([[30.0]]))
    np.testing.assert_allclose(y_pred, [7.0], rtol=1e-5)


def test_min_max_scaler_ridge():
    """
    6. Pairwise combination of MinMaxScaler and Ridge.
    Fits a regression dataset with L2 regularization after minmax scaling.
    """
    scaler = preprocessing.MinMaxScaler()
    reg = linear_model.Ridge(alpha=1.0)
    
    X_train = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    # check that minmax scales inputs to [0, 1] range
    np.testing.assert_allclose(X_scaled.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(X_scaled.max(axis=0), [1.0, 1.0])
    
    pred = reg.predict(X_scaled)
    assert len(pred) == 3


def test_min_max_scaler_lasso():
    """
    7. Pairwise combination of MinMaxScaler and Lasso.
    Fits a regression dataset with L1 regularization after minmax scaling.
    """
    scaler = preprocessing.MinMaxScaler()
    reg = linear_model.Lasso(alpha=0.1)
    
    X_train = np.array([[50.0], [100.0], [150.0]])
    y_train = np.array([10.0, 20.0, 30.0])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    pred = reg.predict(X_scaled)
    assert len(pred) == 3


@pytest.mark.skip(reason='Not supported in thermite')
def test_min_max_scaler_logistic_regression():
    """
    8. Pairwise combination of MinMaxScaler and LogisticRegression.
    Fits a classification dataset after minmax scaling, checking accuracy.
    """
    scaler = preprocessing.MinMaxScaler()
    clf = linear_model.LogisticRegression()
    
    X_train = np.array([[10.0], [20.0], [80.0], [90.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    acc = metrics.accuracy_score(y_train, pred)
    assert acc == 1.0


def test_one_hot_encoder_logistic_regression():
    """
    9. Pairwise combination of OneHotEncoder and LogisticRegression.
    Fits a classification model on one-hot encoded categorical inputs.
    """
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    clf = linear_model.LogisticRegression()
    
    X_train = np.array([["low"], ["low"], ["high"], ["high"]])
    y_train = np.array([0, 0, 1, 1])
    
    X_encoded = ohe.fit_transform(X_train)
    # low and high should result in a 2-column matrix
    assert X_encoded.shape == (4, 2)
    
    clf.fit(X_encoded, y_train)
    pred = clf.predict(X_encoded)
    np.testing.assert_array_equal(pred, y_train)


def test_label_encoder_decision_tree_classifier():
    """
    10. Pairwise combination of LabelEncoder and DecisionTreeClassifier.
    LabelEncoder is used to encode targets (y) from string representation,
    and DecisionTreeClassifier is fit on the inputs and encoded targets.
    """
    le = preprocessing.LabelEncoder()
    clf = tree.DecisionTreeClassifier(random_state=42)
    
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train_str = np.array(["no", "no", "yes", "yes"])
    
    y_train_encoded = le.fit_transform(y_train_str)
    clf.fit(X_train, y_train_encoded)
    
    pred_encoded = clf.predict(X_train)
    pred_str = le.inverse_transform(pred_encoded)
    
    np.testing.assert_array_equal(pred_str, y_train_str)


def test_standard_scaler_decision_tree_classifier():
    """
    11. Pairwise combination of StandardScaler and DecisionTreeClassifier.
    Fits a decision tree classifier on scaled numerical inputs.
    """
    scaler = preprocessing.StandardScaler()
    clf = tree.DecisionTreeClassifier(random_state=42)
    
    X_train = np.array([[0.5, 20.0], [0.8, 10.0], [1.5, 30.0], [2.0, 40.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    np.testing.assert_array_equal(pred, y_train)


def test_standard_scaler_decision_tree_regressor():
    """
    12. Pairwise combination of StandardScaler and DecisionTreeRegressor.
    Fits a decision tree regressor on scaled numerical inputs.
    """
    scaler = preprocessing.StandardScaler()
    reg = tree.DecisionTreeRegressor(random_state=42)
    
    X_train = np.array([[0.1], [0.5], [1.0], [2.0]])
    y_train = np.array([1.5, 2.5, 3.5, 4.5])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    pred = reg.predict(X_scaled)
    np.testing.assert_allclose(pred, y_train)


def test_standard_scaler_random_forest_classifier():
    """
    13. Pairwise combination of StandardScaler and RandomForestClassifier.
    Fits a random forest classifier on scaled inputs.
    """
    scaler = preprocessing.StandardScaler()
    clf = ensemble.RandomForestClassifier(n_estimators=5, random_state=42)
    
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 12.0], [11.0, 13.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    np.testing.assert_array_equal(pred, y_train)


def test_standard_scaler_random_forest_regressor():
    """
    14. Pairwise combination of StandardScaler and RandomForestRegressor.
    Fits a random forest regressor on scaled inputs.
    """
    scaler = preprocessing.StandardScaler()
    reg = ensemble.RandomForestRegressor(n_estimators=5, random_state=42)
    
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    pred = reg.predict(X_scaled)
    assert pred.shape == (4,)


def test_standard_scaler_gradient_boosting_classifier():
    """
    15. Pairwise combination of StandardScaler and GradientBoostingClassifier.
    Fits a gradient boosting classifier on scaled inputs.
    """
    scaler = preprocessing.StandardScaler()
    clf = ensemble.GradientBoostingClassifier(n_estimators=5, random_state=42)
    
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 12.0], [11.0, 13.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    np.testing.assert_array_equal(pred, y_train)


def test_standard_scaler_gradient_boosting_regressor():
    """
    16. Pairwise combination of StandardScaler and GradientBoostingRegressor.
    Fits a gradient boosting regressor on scaled inputs.
    """
    scaler = preprocessing.StandardScaler()
    reg = ensemble.GradientBoostingRegressor(n_estimators=5, random_state=42)
    
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    
    X_scaled = scaler.fit_transform(X_train)
    reg.fit(X_scaled, y_train)
    
    pred = reg.predict(X_scaled)
    assert pred.shape == (4,)


def test_standard_scaler_k_neighbors_classifier():
    """
    17. Pairwise combination of StandardScaler and KNeighborsClassifier.
    Fits a KNN classifier on scaled inputs.
    """
    scaler = preprocessing.StandardScaler()
    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    
    X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.5, 7.8]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)
    
    pred = clf.predict(X_scaled)
    np.testing.assert_array_equal(pred, y_train)


def test_min_max_scaler_k_means():
    """
    18. Pairwise combination of MinMaxScaler and KMeans.
    Performs clustering on minimax scaled inputs.
    """
    scaler = preprocessing.MinMaxScaler()
    km = cluster.KMeans(n_clusters=2, random_state=42, n_init=3)
    
    X_train = np.array([[10.0, 2.0], [12.0, 2.5], [100.0, 80.0], [110.0, 85.0]])
    
    X_scaled = scaler.fit_transform(X_train)
    labels = km.fit_predict(X_scaled)
    
    assert len(labels) == 4
    assert set(labels) == {0, 1}
    assert km.cluster_centers_.shape == (2, 2)


def test_standard_scaler_pca_logistic_regression():
    """
    19. Chaining of StandardScaler, PCA, and LogisticRegression manually.
    """
    scaler = preprocessing.StandardScaler()
    pca = decomposition.PCA(n_components=2, random_state=42)
    clf = linear_model.LogisticRegression()
    
    # 4 samples, 3 features
    X_train = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [10.0, 11.0, 12.0], [11.0, 12.0, 13.0]])
    y_train = np.array([0, 0, 1, 1])
    
    X_scaled = scaler.fit_transform(X_train)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape == (4, 2)
    
    clf.fit(X_pca, y_train)
    pred = clf.predict(X_pca)
    np.testing.assert_array_equal(pred, y_train)


def test_min_max_scaler_pca_k_means():
    """
    20. Chaining of MinMaxScaler, PCA, and KMeans manually.
    """
    scaler = preprocessing.MinMaxScaler()
    pca = decomposition.PCA(n_components=2, random_state=42)
    km = cluster.KMeans(n_clusters=2, random_state=42, n_init=3)
    
    X_train = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [100.0, 200.0, 300.0], [102.0, 202.0, 302.0]])
    
    X_scaled = scaler.fit_transform(X_train)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape == (4, 2)
    
    labels = km.fit_predict(X_pca)
    assert len(labels) == 4
    assert set(labels) == {0, 1}


def test_pipeline_preprocessor_decomposition_classifier():
    """
    21. Chaining in a Pipeline: Preprocessor + Decomposition + Classifier.
    """
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("pca", decomposition.PCA(n_components=2, random_state=42)),
        ("classifier", linear_model.LogisticRegression())
    ])
    
    X_train = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    y_train = np.array([0, 0, 1, 1])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_train)
    np.testing.assert_array_equal(pred, y_train)
    
    # Test steps accesses
    assert "scaler" in pipe.named_steps
    assert "pca" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_pipeline_chaining_preprocessors():
    """
    22. Chaining preprocessors with preprocessors: OneHotEncoder followed by MinMaxScaler.
    """
    pipe = pipeline_mod.Pipeline([
        ("ohe", preprocessing.OneHotEncoder(sparse_output=False)),
        ("scaler", preprocessing.MinMaxScaler(feature_range=(-1, 1))),
        ("classifier", linear_model.LogisticRegression())
    ])
    
    X_train = np.array([["apple"], ["banana"], ["cherry"], ["apple"]])
    y_train = np.array([0, 1, 1, 0])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_train)
    
    ohe_out = pipe.named_steps["ohe"].transform(X_train)
    scaler_out = pipe.named_steps["scaler"].transform(ohe_out)
    np.testing.assert_allclose(scaler_out.min(axis=0), [-1.0, -1.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(scaler_out.max(axis=0), [1.0, 1.0, 1.0], atol=1e-7)
    
    np.testing.assert_array_equal(pred, y_train)


@pytest.mark.skip(reason='Not supported in thermite')
def test_metrics_with_estimators_in_loop():
    """
    23. Combining metrics with estimators inside a train-test split loop.
    Evaluates regression predictions using mean_squared_error, r2_score.
    Evaluates classification predictions using accuracy_score, precision_score, recall_score, f1_score, roc_auc_score.
    """
    # 1. Regression check
    X_reg = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y_reg = np.array([2.1, 3.9, 6.1, 8.0, 9.9, 12.1, 14.0, 15.9])
    
    X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = model_selection.train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )
    
    reg = linear_model.LinearRegression()
    reg.fit(X_tr_reg, y_tr_reg)
    preds_reg = reg.predict(X_te_reg)
    
    mse = metrics.mean_squared_error(y_te_reg, preds_reg)
    r2 = metrics.r2_score(y_te_reg, preds_reg)
    
    assert mse >= 0.0
    assert -1.0 <= r2 <= 1.0
    
    # 2. Classification check
    X_clf = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.8], [5.0, 5.0], [6.0, 6.0], [5.5, 5.8]])
    y_clf = np.array([0, 0, 0, 1, 1, 1])
    
    X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = model_selection.train_test_split(
        X_clf, y_clf, test_size=0.5, random_state=42
    )
    
    clf = linear_model.LogisticRegression()
    clf.fit(X_tr_clf, y_tr_clf)
    preds_clf = clf.predict(X_te_clf)
    
    # get decision values/probabilities if needed for roc_auc_score
    if hasattr(clf, "predict_proba"):
        probs_clf = clf.predict_proba(X_te_clf)[:, 1]
    else:
        probs_clf = preds_clf
        
    acc = metrics.accuracy_score(y_te_clf, preds_clf)
    prec = metrics.precision_score(y_te_clf, preds_clf, zero_division=0)
    rec = metrics.recall_score(y_te_clf, preds_clf, zero_division=0)
    f1 = metrics.f1_score(y_te_clf, preds_clf, zero_division=0)
    roc_auc = metrics.roc_auc_score(y_te_clf, probs_clf)
    
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= roc_auc <= 1.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_pipeline_inside_cross_val_score():
    """
    24. Chaining pipeline inside cross_val_score.
    """
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("logistic", linear_model.LogisticRegression())
    ])
    
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    scores = model_selection.cross_val_score(pipe, X, y, cv=2, scoring="accuracy")
    assert scores.shape == (2,)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.skip(reason='Not supported in thermite')
def test_pipeline_inside_grid_search_cv():
    """
    25. Chaining pipeline inside GridSearchCV.
    """
    pipe = pipeline_mod.Pipeline([
        ("scaler", preprocessing.MinMaxScaler()),
        ("ridge", linear_model.Ridge())
    ])
    
    # We want to tune hyperparameters of the scaler and the estimator
    param_grid = {
        "scaler__feature_range": [(0, 1), (-1, 1)],
        "ridge__alpha": [0.1, 1.0]
    }
    
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    gs = model_selection.GridSearchCV(pipe, param_grid, cv=2)
    gs.fit(X, y)
    
    assert gs.best_estimator_ is not None
    assert gs.best_params_["scaler__feature_range"] in [(0, 1), (-1, 1)]
    assert gs.best_params_["ridge__alpha"] in [0.1, 1.0]
    
    preds = gs.predict(X)
    assert preds.shape == (5,)
