# Thermite E2E Test Infrastructure

## Test Philosophy
Thermite's E2E test suite is **opaque-box** and **requirement-driven**. It focuses on validating that the library acts as a seamless, drop-in replacement for `scikit-learn` from a Python API and functional output perspective. 

To ensure test correctness, the test suite supports a backend switcher controlled by the `USE_SKLEARN` environment variable. When set to `1` or `true`, the test suite executes against the standard `scikit-learn` package. This allows us to validate the correctness of the E2E tests themselves before running them against `thermite`. When unset or set to `0` or `false`, the tests execute against the `thermite` package.

---

## Feature Inventory
The E2E test suite covers all **29 features** specified in the Thermite requirements:

### 1. Preprocessing (`thermite.preprocessing`)
- `StandardScaler`
- `MinMaxScaler`
- `LabelEncoder`
- `OneHotEncoder`

### 2. Linear Models (`thermite.linear_model`)
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LogisticRegression`

### 3. Tree-Based & Ensembles (`thermite.tree`, `thermite.ensemble`)
- `DecisionTreeClassifier`
- `DecisionTreeRegressor`
- `RandomForestClassifier`
- `RandomForestRegressor`
- `GradientBoostingClassifier`
- `GradientBoostingRegressor`

### 4. Clustering (`thermite.cluster`)
- `KMeans`
- `DBSCAN`

### 5. Decomposition (`thermite.decomposition`)
- `PCA`

### 6. Neighbors (`thermite.neighbors`)
- `KNeighborsClassifier`

### 7. Model Selection (`thermite.model_selection`)
- `train_test_split`
- `cross_val_score`
- `GridSearchCV`

### 8. Pipeline (`thermite.pipeline`)
- `Pipeline`

### 9. Metrics (`thermite.metrics`)
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `roc_auc_score`
- `mean_squared_error`
- `r2_score`

---

## Test Architecture

### Directory Layout
```
/Users/kartavyadikshit/Projects/Thermite/
├── tests/
│   ├── conftest.py           # Backend switcher and module resolution helper
│   ├── test_infra_check.py   # Basic verification of switcher setup
│   ├── ...                   # Future E2E tier tests
```

### The Backend Switcher (`tests/conftest.py`)
All E2E test files must not import modules directly from `sklearn` or `thermite`. Instead, they import the `get_module` helper from `tests.conftest`:

```python
from tests.conftest import get_module

# Dynamically import the required module based on the USE_SKLEARN environment variable
linear_model = get_module("linear_model")
metrics = get_module("metrics")

# Use classes/functions as usual
model = linear_model.LogisticRegression()
```

### How to Run Tests

Running against `scikit-learn` (for test validation):
```bash
PYTHONPATH=. USE_SKLEARN=1 pytest tests/
```

Running against `thermite` (for implementation verification):
```bash
PYTHONPATH=. pytest tests/
```

---

## Real-World Application Scenarios (Tier 4)
We have planned **5 E2E scenarios** representing realistic application-level workloads:

1. **End-to-End Predictive Maintenance Pipeline**
   - **Goal**: Predict machine component failures from sensor logs.
   - **Features Used**: `Pipeline`, `MinMaxScaler`, `StandardScaler`, `RandomForestClassifier`, `KNeighborsClassifier`, `train_test_split`, `accuracy_score`, `f1_score`.
   - **Workflow**: Load synthetic numerical time-series sensors, split train/test, apply scaling, train/predict with classifiers, and report validation metrics.

2. **Customer Segmentation and Profiling**
   - **Goal**: Segment users based on behavioral and transaction metrics.
   - **Features Used**: `MinMaxScaler`, `PCA`, `KMeans`, `Pipeline`.
   - **Workflow**: Process transaction and demographic data, perform dimensionality reduction via PCA to project to 2D/3D space, cluster with KMeans, and evaluate cluster cohesion.

3. **Credit Risk Assessment with Grid Search**
   - **Goal**: Model loan default risk using categorical and numerical features.
   - **Features Used**: `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `GridSearchCV`, `Pipeline`, `precision_score`, `recall_score`, `roc_auc_score`.
   - **Workflow**: Scale numeric and one-hot encode categorical customer data, build pipeline with LogisticRegression, optimize hyper-parameters via GridSearchCV, and score with precision/recall/ROC AUC.

4. **House Price Forecasting and Feature Selection**
   - **Goal**: Predict residential house prices and identify key pricing features.
   - **Features Used**: `StandardScaler`, `LinearRegression`, `Ridge`, `Lasso`, `train_test_split`, `mean_squared_error`, `r2_score`.
   - **Workflow**: Scale Ames housing data, apply Lasso/Ridge regression for feature selection/regularization, train final estimator, and evaluate with MSE and R^2.

5. **Multi-Model Ensemble Classifier for Fraud Detection**
   - **Goal**: Evaluate multiple models on highly imbalanced transaction data.
   - **Features Used**: `MinMaxScaler`, `OneHotEncoder`, `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`, `cross_val_score`, `f1_score`, `precision_score`, `recall_score`.
   - **Workflow**: Preprocess features, run K-fold cross-validation (`cross_val_score`) comparing Random Forest, Gradient Boosting, and Logistic Regression models, select the best model, and report detailed classification metrics.

---

## Coverage Thresholds

To ensure a robust, high-fidelity test suite, the following coverage counts are enforced:

| Testing Tier | Description | Target Count |
|--------------|-------------|--------------|
| **Tier 1** | Feature Coverage (Happy-path tests) | >= 5 test cases per feature/algorithm (**>=145** cases total) |
| **Tier 2** | Boundary & Corner Cases (Extreme inputs, dimensions, edge values) | >= 5 test cases per feature/algorithm (**>=145** cases total) |
| **Tier 3** | Cross-Feature Combinations (Pairwise estimator & preprocessor chaining) | **>=20** combination cases |
| **Tier 4** | Real-World Scenarios (End-to-end business workflows) | **>=5** comprehensive integration tests |
| **Total** | **E2E test suite coverage** | **>=315** test cases |
