## 2026-07-12T11:24:09Z
You are explorer_m1_alg_3.
Your working directory is `/Users/kartavyadikshit/Projects/Thermite/.agents/explorer_m1_alg_3`.
Your parent is Milestone 1 Orchestrator (Conversation ID: 4f539ea2-b299-4cac-afb7-27d4a5777e72).
Your task is to design the implementation of the algorithms and utilities for Milestones M1-2 and M1-3 in `/Users/kartavyadikshit/Projects/Thermite/.agents/sub_orch_m1/SCOPE.md`.
Specifically, design:
1. `train_test_split`: Exposes function `train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)`. In Rust core, write the shuffle/split logic using ndarray, rand (SmallRng for seed reproducibility). Wrap via PyO3 to accept and return numpy arrays.
2. `StandardScaler`: Rust-side struct storing mean, var, scale, n_samples_seen. Expose fit, transform, fit_transform, inverse_transform. Use Rayon for parallel mean/variance calculations.
3. `MinMaxScaler`: Rust-side struct storing data_min, data_max, scale, min. Expose fit, transform, fit_transform, inverse_transform.
4. `LabelEncoder`: Exposes fit, transform, fit_transform, inverse_transform for 1D arrays of integers or strings.
5. `OneHotEncoder`: Exposes fit, transform, fit_transform, inverse_transform for 2D categorical features (integers or strings).

Read `/Users/kartavyadikshit/Projects/Thermite/PROJECT.md` and the handoff of Milestone 1-1.
Do NOT write any code directly in the codebase.
Write your analysis, recommendations, and recommended Rust/Python code templates to `/Users/kartavyadikshit/Projects/Thermite/.agents/explorer_m1_alg_3/handoff.md` and update `/Users/kartavyadikshit/Projects/Thermite/.agents/explorer_m1_alg_3/progress.md`.
Once complete, send a message back with your handoff.md path.
