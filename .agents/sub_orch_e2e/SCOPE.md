# Scope: E2E Testing Track

## Architecture
- Build a requirement-driven, opaque-box E2E test suite.
- It must run against the installed `thermite` package (after `pip install -e .`).
- Test layout should be in `tests/` directory.
- It must not depend on internal Rust implementation details, only on python-facing API compatibility with scikit-learn.
- Test runner: `pytest`.
- Output: publishes `TEST_READY.md` when all tests are ready (expected to fail until the implementation is complete).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E-1 | Test Infra Setup | Set up pytest, test directory structure, and basic runner script. | None | DONE |
| E2E-2 | Tier 1 Feature Coverage | Write >=5 happy-path test cases per algorithm/feature. | E2E-1 | DONE |
| E2E-3 | Tier 2 Boundary Cases | Write >=5 edge and boundary test cases per feature. | E2E-2 | DONE |
| E2E-4 | Tier 3 Cross-Feature | Write pairwise cross-feature combination test cases. | E2E-3 | DONE |
| E2E-5 | Tier 4 Real-World | Write realistic application workload scenarios. | E2E-4 | IN_PROGRESS (94c57302-222f-475e-ad2e-8f8453ecc3bc) |
| E2E-6 | Publish TEST_READY | Generate and write `TEST_READY.md` at project root. | E2E-5 | PLANNED |

## Interface Contracts
- Standard Python entry points: `import thermite`
- API signatures matching scikit-learn exactly.
