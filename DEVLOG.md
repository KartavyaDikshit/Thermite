# 🔥 Thermite — Development Log

> **Project**: Thermite — A Rust-accelerated, scikit-learn-compatible ML library for Python
> **Repository**: https://github.com/KartavyaDikshit/Thermite
> **Started**: 2026-07-12
> **Status**: 🟡 Planning & Architecture

---

## Roadmap

### Phase 1: Foundation (Current)
- [x] Research landscape & identify opportunity
- [x] Choose project: Rust-accelerated scikit-learn
- [x] Initialize repository & GitHub sync
- [ ] Define architecture & module structure
- [ ] Set up Rust workspace (Cargo.toml)
- [ ] Set up Python package (pyproject.toml + maturin)
- [ ] Implement core linear algebra primitives
- [ ] Implement `StandardScaler` (first preprocessing module)
- [ ] Implement `train_test_split` utility
- [ ] First benchmark vs scikit-learn

### Phase 2: Core Algorithms
- [ ] `LinearRegression` / `Ridge` / `Lasso`
- [ ] `LogisticRegression`
- [ ] `DecisionTreeClassifier` / `DecisionTreeRegressor`
- [ ] `RandomForestClassifier` / `RandomForestRegressor`
- [ ] `GradientBoostingClassifier` / `GradientBoostingRegressor`
- [ ] `KMeans` clustering
- [ ] `PCA` dimensionality reduction
- [ ] `KNeighborsClassifier`

### Phase 3: Ecosystem
- [ ] Full preprocessing module (encoders, imputers, scalers)
- [ ] Model selection (cross_val_score, GridSearchCV)
- [ ] Pipeline support (sklearn-compatible Pipeline)
- [ ] Metrics module (accuracy, precision, recall, F1, ROC-AUC)
- [ ] Serialization (save/load models)

### Phase 4: Polish & Launch
- [ ] Comprehensive benchmarks suite
- [ ] Documentation site
- [ ] PyPI package (`pip install thermite-ml`)
- [ ] Blog post / launch announcement
- [ ] CI/CD pipeline

---

## Decision Log

### 2026-07-12 — Project Selection
**Decision**: Build a Rust-accelerated scikit-learn replacement
**Rationale**:
- scikit-learn has 40M+ monthly PyPI downloads — largest untapped Rust acceleration opportunity
- Follows the proven Polars/Pydantic/uv playbook (Rust core + Python API via PyO3)
- `linfa` exists as pure Rust ML but has NO Python-facing drop-in replacement
- Clear startup/acquisition path — OpenAI acquired Astral, Cloudflare acquired VoidZero for similar projects
- Rust's parallelism gives massive wins on CPU-bound ML algorithms (trees, ensembles, clustering)

**Alternatives Considered**:
| Option | Why Not |
|--------|---------|
| Rust CI/CD Engine | Ecosystem lock-in too strong; needs full plugin compatibility |
| Rust Airflow | Dagster/Prefect iterating fast; narrowing window |
| Rust pytest | Plugin ecosystem is the moat, not runner speed |
| TS Type Checker | stc failed; 5+ year effort for a large team |
| Rust Load Testing | k6 (Go) already fast enough |

### 2026-07-12 — Architecture Decisions
**Tech Stack**:
- **Rust core**: Workspace with multiple crates (`thermite-core`, `thermite-linalg`, `thermite-io`)
- **Python bindings**: PyO3 + maturin for build/packaging
- **API design**: sklearn-compatible — `fit()`, `predict()`, `transform()`, `fit_transform()`
- **Data format**: Accept numpy arrays, return numpy arrays (via numpy PyO3 bindings)
- **Parallelism**: Rayon for data parallelism within algorithms

---

## Work Log

### 2026-07-12 13:15 — Project Kickoff
- Completed comprehensive research on the Rust rewrite ecosystem
- Mapped 50+ existing Rust rewrites, identified gaps
- Selected scikit-learn acceleration as the target project
- Initialized git repository
- Set up GitHub remote: https://github.com/KartavyaDikshit/Thermite.git
- Created DEVLOG.md (this file)
- Created initial README.md
- Beginning teamwork prompt crafting for multi-agent implementation

---

*This log is updated as work progresses. Each entry includes timestamp, what was done, and any decisions made.*
