# Welcome to the Thermite ML Wiki!

**Thermite ML** is a blazing-fast, Rust-accelerated machine learning library for Python. It is designed to be a drop-in replacement for scikit-learn, bringing massive performance improvements, zero-copy data ingestion, and out-of-the-box hardware acceleration to your ML pipelines.

## Navigation

* **[API Reference](API.md):** Complete guide to the supported algorithms, enterprise capabilities (federated learning, RAG, etc.), and Python module structures.
* **[Architecture](ARCHITECTURE.md):** Deep dive into the Rust core (`thermite-core`), GPU backend (`thermite-gpu`), and FFI bindings.
* **[Performance Metrics](METRICS.md):** Detailed benchmarking results against scikit-learn across standard algorithms.

## Key Highlights

- **Hardware Acceleration:** Native WebGPU/CUDA acceleration for ensembles and matrix math without requiring heavy CUDA toolkits.
- **Zero-Copy Polars Integration:** Feed Apache Arrow and `polars` DataFrames without memory duplication.
- **True Parallelism:** Bypasses Python's Global Interpreter Lock (GIL) via Rust's Rayon threading.

To install the latest version:
```bash
pip install thermite-ml==2.6.6
```
