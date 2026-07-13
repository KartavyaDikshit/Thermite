# Thermite ML - Architecture

Thermite ML is composed of three main layers:

1. `thermite-core` (Rust): The mathematical and algorithmic engine. Uses `ndarray` for matrix operations, `rayon` for CPU parallelism, and `bincode`/`serde` for serialization.
2. `thermite-gpu` (Rust): The hardware acceleration backend using `wgpu`. Compiles and dispatches WGSL shaders for cross-platform (Metal/Vulkan/DX12) GPU computation.
3. `thermite-binding` (Rust/PyO3): The FFI layer linking Python with Rust. Implements zero-copy memory transfers and the Global Interpreter Lock (GIL) release mechanisms.
4. `thermite` (Python): The scikit-learn compatible frontend. Wraps the bindings and handles validation, model cards, and fallback mechanisms.

## Subsystems

- **Tree & Ensembles**: Built using a highly parallelized histogram-based approach. Native categorical support prevents One-Hot Encoding overhead. Predictions are aggregated via GPU (if enabled) for maximal throughput.
- **Linear Models**: Built on optimized BLAS operations with matrix inversion and SGD implementations. Supports L1/L2 regularizations and sparsity natively.
- **Hardware Acceleration**: The `DeviceKind` abstraction routes `device='gpu'` requests to the `thermite-gpu` crate, moving massive `NxM` operations to VRAM without CUDA requirements.
- **Advanced Nodes**: 
  - `ParameterServer` (Federated Learning): Averages gradients from edge devices safely.
  - `TLearner` (Causal): Estimates treatment effects using dual regressors.
  - `VectorStore` (RAG): High-speed Euclidean proxy for local embeddings search.
