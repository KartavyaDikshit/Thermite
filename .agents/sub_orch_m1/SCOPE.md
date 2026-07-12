# Scope: Milestone 1 - Foundation & Preprocessing

## Architecture
- Set up Cargo workspace and Maturin compilation structure.
- Define PyO3 bindings structure.
- Create Python package folder structure.
- Implement the following algorithms/utils:
  - Preprocessing: `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`
  - Model Selection utility: `train_test_split`

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1-1 | Build Setup & Packaging | Create `Cargo.toml`, `pyproject.toml`, and Python package structure. Verify maturin develop compiles and installs. | None | DONE |
| M1-2 | train_test_split utility | Implement `train_test_split` in Rust core and wrap in Python. | M1-1 | PLANNED |
| M1-3 | Scalers & Encoders | Implement `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder` in Rust/Python. | M1-1 | PLANNED |
| M1-4 | Verification & Handoff | Verify all unit and basic integration tests pass. Deliver handoff. | M1-2, M1-3 | PLANNED |

## Interface Contracts
- Input array types: numpy arrays or list-like shapes convertible to numpy arrays.
- Return values: numpy arrays.
- Classes: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder.
- Functions: train_test_split.
