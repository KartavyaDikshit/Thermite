## 2026-07-12T17:48:13Z
Please implement NaN / Missing Data Native Support in Thermite.
Your working directory is /Users/kartavyadikshit/Projects/Thermite/.agents/worker_f1_nan_support.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Tasks:
1. **Decision Trees** (`crates/thermite-core/src/tree.rs`):
   - Modify `TreeNode` struct to include `pub nan_go_left: bool`.
   - Update split evaluation functions (`find_best_classification_split` and `find_best_regression_split`):
     - Separate sample indices into those with NaN values in the feature column and those with non-NaN values.
     - Sort only the non-NaN indices by feature values.
     - When evaluating a split threshold on the sorted non-NaN indices, compute the Gini/MSE gain for two scenarios: routing all NaNs to the left child, or routing all NaNs to the right child.
     - Choose the feature, threshold, and `nan_go_left` direction that maximizes the split gain.
   - Update `predict_proba_single` and `predict_single` in trees: if the feature value for the node is NaN, route the sample based on `node.nan_go_left`.
   - Ensure the ensemble models (RandomForest, GradientBoosting) work correctly with this tree backend.
2. **Linear Models** (`crates/thermite-core/src/linear_model.rs`):
   - Add `impute_values: Option<Vec<f64>>` to linear model structs (LinearRegression, Ridge, Lasso, LogisticRegression).
   - Relax `check_finite_2d` to reject infinite values but allow NaN values.
   - In `fit()`, compute the mean of each column of feature matrix X ignoring NaNs. Store these column means in `impute_values`. Replace NaNs in X with the corresponding column mean before training.
   - In `predict()` and `predict_proba()`, replace NaNs in input feature matrix X with the stored column means in `impute_values`.
3. **GridSearch Parameter Fix** (`thermite/model_selection.py`):
   - Fix the `base_params` extraction issue. Do not collect callable methods (like `partial_fit`) as parameters. Filter out functions or callables in the `dir(self.estimator)` loop using `callable(val)` or similar checks.
4. **Verification**:
   - Create a Python test script `tests/test_nan_support.py` that generates a dummy dataset containing `np.nan` values, trains and predicts using `DecisionTreeClassifier` and `LogisticRegression`, and verifies that they train and predict successfully without exceptions and achieve >90% accuracy.
   - Run `cargo test` and `pytest tests/test_nan_support.py` to verify the build and tests succeed.
   - Write a detailed `handoff.md` with your changes and test outputs in your directory.
