// thermite-gpu/src/lib.rs
// GPU acceleration backend for Thermite.
//
// Architecture:
//   - DeviceKind enum: controls dispatch (Cpu | Gpu)
//   - GpuContext: wraps wgpu device + queue + shaders (feature-gated)
//   - GpuTensor: a contiguous f32 buffer on the GPU (feature-gated)
//   - Fallback: when the `wgpu` feature is not compiled in, every call
//     transparently falls back to the CPU path in thermite-core.
//
// Usage from Python (via binding layer in thermite-binding):
//   clf = RandomForestClassifier(device='cuda')   # => DeviceKind::Gpu
//   clf = RandomForestClassifier(device='cpu')    # => DeviceKind::Cpu

#![allow(unused_variables, dead_code)]

// =====================================================================
// Device abstraction (always compiled)
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

impl DeviceKind {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gpu" | "cuda" | "wgpu" => DeviceKind::Gpu,
            _ => DeviceKind::Cpu,
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceKind::Gpu)
    }
}

// =====================================================================
// Matrix multiply dispatch
// Signature is identical regardless of device: (m x k) @ (k x n) -> (m x n)
// Both inputs and output use row-major f32 (GPU-native format).
// =====================================================================

/// General matrix multiply: C = A @ B  (row-major, f32)
/// On CPU path: uses ndarray + rayon (safe, always available).
/// On GPU path: dispatches to wgpu compute shader (feature-gated).
pub fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    device: DeviceKind,
) -> Vec<f32> {
    #[cfg(feature = "wgpu")]
    if device == DeviceKind::Gpu {
        return gpu::matmul_gpu(a, b, m, k, n);
    }
    cpu::matmul_cpu(a, b, m, k, n)
}

/// Sigmoid activation: element-wise 1/(1+exp(-x))
pub fn sigmoid(x: &[f32], device: DeviceKind) -> Vec<f32> {
    #[cfg(feature = "wgpu")]
    if device == DeviceKind::Gpu {
        return gpu::sigmoid_gpu(x);
    }
    cpu::sigmoid_cpu(x)
}

/// Dot product of two flat vectors
pub fn dot(a: &[f32], b: &[f32], device: DeviceKind) -> f32 {
    #[cfg(feature = "wgpu")]
    if device == DeviceKind::Gpu {
        return gpu::dot_gpu(a, b);
    }
    cpu::dot_cpu(a, b)
}

/// Majority vote across `n_estimators` predictions.
/// `predictions` is a flat row-major array: [estimator_0_pred_0, estimator_0_pred_1, ..., estimator_k_pred_n]
/// Outer dimension = n_estimators, inner dimension = n_samples.
/// Returns Vec<f32> of length n_samples containing the majority class (as f32 bits = class label bits).
pub fn ensemble_majority_vote(
    predictions: &[f32],
    n_estimators: usize,
    n_samples: usize,
    device: DeviceKind,
) -> Vec<f32> {
    // GPU reduction for this case: large n_estimators * n_samples
    // For now, always use highly-optimised parallel CPU path.
    // wgpu kernel can be plugged in identically to matmul_gpu.
    let _ = device; // device reserved for future GPU dispatch
    cpu::majority_vote_cpu(predictions, n_estimators, n_samples)
}

/// Row-wise average across `n_estimators` predictions.
/// `predictions` layout: [est_0_sample_0, est_0_sample_1, ..., est_k_sample_n]
pub fn ensemble_row_mean(
    predictions: &[f32],
    n_estimators: usize,
    n_samples: usize,
    device: DeviceKind,
) -> Vec<f32> {
    let _ = device;
    cpu::row_mean_cpu(predictions, n_estimators, n_samples)
}

// =====================================================================
// CPU fallback (always compiled, zero dependencies beyond std)
// =====================================================================
mod cpu {
    use rayon::prelude::*;

    pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c_row[j] = sum;
            }
        });
        c
    }

    pub fn sigmoid_cpu(x: &[f32]) -> Vec<f32> {
        x.par_iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    pub fn dot_cpu(a: &[f32], b: &[f32]) -> f32 {
        a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Majority vote: predictions[est * n_samples + sample] -> winner per sample.
    pub fn majority_vote_cpu(predictions: &[f32], n_estimators: usize, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .into_par_iter()
            .map(|s| {
                let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
                for e in 0..n_estimators {
                    let v = predictions[e * n_samples + s];
                    *counts.entry(v.to_bits()).or_insert(0) += 1;
                }
                let best_bits = counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
                f32::from_bits(best_bits)
            })
            .collect()
    }

    /// Row-wise mean: predictions[est * n_samples + sample] -> mean per sample.
    pub fn row_mean_cpu(predictions: &[f32], n_estimators: usize, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .into_par_iter()
            .map(|s| {
                let sum: f32 = (0..n_estimators).map(|e| predictions[e * n_samples + s]).sum();
                sum / n_estimators as f32
            })
            .collect()
    }
}

// =====================================================================
// wgpu GPU backend (compiled only with `--features wgpu`)
// =====================================================================
#[cfg(feature = "wgpu")]
mod gpu {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct MatmulParams {
        m: u32,
        k: u32,
        n: u32,
        _pad: u32,
    }

    // Lazily initialise a wgpu device+queue for the current process.
    // In production, this would be cached in a global OnceLock.
    fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("No GPU adapter found. Thermite GPU requires a wgpu-compatible device.");

            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("thermite-gpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: Default::default(),
                    },
                    None,
                )
                .await
                .expect("Failed to create wgpu device")
        })
    }

    // WGSL compute shader for GEMM: C[m,n] = sum_k A[m,k]*B[k,n]
    const MATMUL_SHADER: &str = r"
        struct Params { m: u32, k: u32, n: u32, _pad: u32 }
        @group(0) @binding(0) var<storage, read>       A      : array<f32>;
        @group(0) @binding(1) var<storage, read>       B      : array<f32>;
        @group(0) @binding(2) var<storage, read_write> C      : array<f32>;
        @group(0) @binding(3) var<uniform>             params : Params;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.x;
            let col = gid.y;
            if row >= params.m || col >= params.n { return; }
            var acc: f32 = 0.0;
            for (var l: u32 = 0u; l < params.k; l++) {
                acc += A[row * params.k + l] * B[l * params.n + col];
            }
            C[row * params.n + col] = acc;
        }
    ";

    const SIGMOID_SHADER: &str = r"
        @group(0) @binding(0) var<storage, read>       X : array<f32>;
        @group(0) @binding(1) var<storage, read_write> Y : array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if i >= arrayLength(&X) { return; }
            Y[i] = 1.0 / (1.0 + exp(-X[i]));
        }
    ";

    pub fn matmul_gpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let (device, queue) = init_gpu();

        let params = MatmulParams { m: m as u32, k: k as u32, n: n as u32, _pad: 0 };
        let out_size = m * n;

        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"), contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B"), contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: (out_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"), contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (out_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"), source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"), layout: Some(&pipeline_layout),
            module: &shader, entry_point: "main", compilation_options: Default::default(), cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                ((m + 15) / 16) as u32,
                ((n + 15) / 16) as u32,
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&buf_c, 0, &buf_staging, 0, (out_size * 4) as u64);
        queue.submit([encoder.finish()]);

        let slice = buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    }

    pub fn sigmoid_gpu(x: &[f32]) -> Vec<f32> {
        // Stub: call CPU fallback for now; full wgpu dispatch follows same pattern as matmul_gpu
        super::cpu::sigmoid_cpu(x)
    }

    pub fn dot_gpu(a: &[f32], b: &[f32]) -> f32 {
        // Stub: call CPU fallback; can be replaced with a wgpu reduction kernel
        super::cpu::dot_cpu(a, b)
    }
}

// =====================================================================
// Tests
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_cpu_identity() {
        // 2x2 identity @ 2x2 identity = 2x2 identity
        let id: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let result = matmul(&id, &id, 2, 2, 2, DeviceKind::Cpu);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_cpu_known_values() {
        let x = vec![0.0f32, f32::INFINITY, f32::NEG_INFINITY];
        let s = sigmoid(&x, DeviceKind::Cpu);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_device_kind_from_string() {
        assert_eq!(DeviceKind::from_string("cuda"), DeviceKind::Gpu);
        assert_eq!(DeviceKind::from_string("GPU"), DeviceKind::Gpu);
        assert_eq!(DeviceKind::from_string("cpu"), DeviceKind::Cpu);
    }
}
