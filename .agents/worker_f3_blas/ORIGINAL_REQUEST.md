## 2026-07-12T17:57:54Z
Please configure Dynamic BLAS/MKL Linkage features in Thermite.
Your working directory is /Users/kartavyadikshit/Projects/Thermite/.agents/worker_f3_blas.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Tasks:
1. **Cargo Configuration** (`crates/thermite-core/Cargo.toml`):
   - Add features for optional hardware BLAS/MKL libraries:
     ```toml
     [features]
     intel-mkl = ["ndarray/blas", "dep:intel-mkl-src"]
     openblas = ["ndarray/blas", "dep:openblas-src"]
     accelerate = ["ndarray/blas", "dep:accelerate-src"]
     ```
   - Add the optional dependencies:
     ```toml
     [dependencies]
     intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-iomp"], optional = true }
     openblas-src = { version = "0.10", default-features = false, features = ["cblas", "system"], optional = true }
     accelerate-src = { version = "0.3", optional = true }
     ```
2. **Linkage Trigger** (`crates/thermite-core/src/lib.rs`):
   - Add the conditional `extern crate` linkage triggers:
     ```rust
     #[cfg(feature = "intel-mkl")]
     extern crate intel_mkl_src;

     #[cfg(feature = "openblas")]
     extern crate openblas_src;

     #[cfg(feature = "accelerate")]
     extern crate accelerate_src;
     ```
3. **Verification**:
   - Run compilation and tests for each feature flag (e.g. `cargo test --features accelerate`, `cargo test --features openblas`, `cargo test --features intel-mkl` or similar, depending on what is supported by the OS and available offline packages). Since the OS is macOS, `accelerate` feature is highly recommended to verify.
   - Verify that there are no compilation or linker errors.
   - Write a detailed `handoff.md` with your changes and test/compilation outputs in your directory.
