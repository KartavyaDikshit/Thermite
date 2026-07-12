## 2026-07-12T11:25:59Z
You are the worker agent for Milestone E2E-3 (Tier 2 Boundary Cases) of Thermite.
Your working directory is `/Users/kartavyadikshit/Projects/Thermite/.agents/worker_e2e_tier2`.

Your task is to write a comprehensive boundary and corner-case test suite (Tier 2) for the 29 features listed in `TEST_INFRA.md`.
You must create the following files in the `tests/` directory:
1. `tests/test_tier2_preprocessing.py`: covering boundary cases for `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`.
2. `tests/test_tier2_linear.py`: covering boundary cases for `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`.
3. `tests/test_tier2_trees.py`: covering boundary cases for `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`.
4. `tests/test_tier2_cluster_decomposition_neighbors.py`: covering boundary cases for `KMeans`, `DBSCAN`, `PCA`, `KNeighborsClassifier`.
5. `tests/test_tier2_model_selection_pipeline.py`: covering boundary cases for `train_test_split`, `cross_val_score`, `GridSearchCV`, `Pipeline`.
6. `tests/test_tier2_metrics.py`: covering boundary cases for `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `mean_squared-error`, `r2_score`.

Requirements for tests:
- Each feature must have at least 5 distinct boundary or corner-case test cases.
  For example:
  - Empty or single-row inputs.
  - Scale with zero variance (all values identical).
  - High dimensions or extremely large/small numerical inputs (check no overflow/underflow or division by zero errors).
  - Categorical encoders with unseen values.
  - Metrics with all zero predictions or targets, causing potential divide-by-zero (e.g. precision/recall/f1) or single class targets (roc_auc_score).
  - Estimators fit on collinear features.
  - DBSCAN with all noise, KMeans with more clusters than samples, PCA with more components than min(samples, features).
  - Empty param grids for GridSearchCV, empty or trivial Pipeline objects.
- All imports of `thermite` modules must be dynamically loaded via `get_module` from `tests.conftest`.
- Verify all boundary cases pass when running with `USE_SKLEARN=1 pytest tests/test_tier2_*.py` using the Homebrew Python environment `.venv_homebrew/bin/pytest`.
- Ensure there are no syntax errors or lints (run Ruff or flake8 on the tests if available).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please update `/Users/kartavyadikshit/Projects/Thermite/.agents/worker_e2e_tier2/progress.md` with your progress. Once complete, write your handoff/completion report to `/Users/kartavyadikshit/Projects/Thermite/.agents/worker_e2e_tier2/handoff.md` and send a message back.
