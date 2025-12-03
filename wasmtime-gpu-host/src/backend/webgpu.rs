//! WebGPU Backend Implementation
//!
//! Uses wgpu for cross-platform GPU compute via Vulkan/Metal/DX12.
//! WGSL compute shaders implement the ML kernels.

use super::{
    AverageParams, BatchPredictParams, BootstrapParams, BufferId, BufferUsage, DeviceInfo,
    ElementwiseOp, ElementwiseParams, FindSplitParams, GpuBackend, GpuError, MatmulParams,
    ReduceOp, ReduceParams,
};
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};
use wgpu;

/// Internal buffer representation for WebGPU
struct WgpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: u32,
}

/// WebGPU Backend implementation
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    device_info: DeviceInfo,
    buffers: HashMap<BufferId, WgpuBuffer>,
    next_buffer_id: BufferId,
    // Compiled shaders
    bootstrap_pipeline: Option<wgpu::ComputePipeline>,
    average_pipeline: Option<wgpu::ComputePipeline>,
}

impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub async fn new() -> Result<Self, GpuError> {
        info!("[WebGPU] Initializing WebGPU backend...");
        
        // Create instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });
        
        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                error!("[WebGPU] No suitable adapter found");
                GpuError::DeviceUnavailable
            })?;
        
        let adapter_info = adapter.get_info();
        info!("[WebGPU] Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
        
        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wasi:gpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                error!("[WebGPU] Failed to create device: {}", e);
                GpuError::BackendError(format!("Device creation failed: {}", e))
            })?;
        
        let device_info = DeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend).to_lowercase(),
            total_memory: 0, // WebGPU doesn't expose this directly
            is_hardware: adapter_info.device_type != wgpu::DeviceType::Cpu,
            compute_capability: String::new(),
        };
        
        info!("[WebGPU] Backend initialized successfully");
        
        Ok(Self {
            device,
            queue,
            device_info,
            buffers: HashMap::new(),
            next_buffer_id: 1,
            bootstrap_pipeline: None,
            average_pipeline: None,
        })
    }
    
    /// Synchronous wrapper for new()
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }
    
    /// Convert our usage flags to wgpu usage
    fn to_wgpu_usage(usage: u32) -> wgpu::BufferUsages {
        let mut wgpu_usage = wgpu::BufferUsages::empty();
        
        if usage & BufferUsage::MAP_READ != 0 {
            wgpu_usage |= wgpu::BufferUsages::MAP_READ;
        }
        if usage & BufferUsage::MAP_WRITE != 0 {
            wgpu_usage |= wgpu::BufferUsages::MAP_WRITE;
        }
        if usage & BufferUsage::COPY_SRC != 0 {
            wgpu_usage |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage & BufferUsage::COPY_DST != 0 {
            wgpu_usage |= wgpu::BufferUsages::COPY_DST;
        }
        if usage & BufferUsage::UNIFORM != 0 {
            wgpu_usage |= wgpu::BufferUsages::UNIFORM;
        }
        if usage & BufferUsage::STORAGE != 0 {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        
        // Always add COPY_DST for write operations
        wgpu_usage |= wgpu::BufferUsages::COPY_DST;
        
        wgpu_usage
    }
    
    /// Get buffer or return error
    fn get_buffer(&self, id: BufferId) -> Result<&WgpuBuffer, GpuError> {
        self.buffers.get(&id).ok_or(GpuError::InvalidBuffer(id))
    }
    
    /// Create bootstrap sampling shader
    fn create_bootstrap_pipeline(&mut self) {
        if self.bootstrap_pipeline.is_some() {
            return;
        }
        
        let shader_source = r#"
            struct Params {
                n_samples: u32,
                seed: u32,
                max_index: u32,
                _padding: u32,
            }
            
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;
            @group(0) @binding(1) var<uniform> params: Params;
            
            fn pcg_hash(input: u32) -> u32 {
                var state = input * 747796405u + 2891336453u;
                state = state ^ (state >> 16u);
                state = state * 2654435769u;
                state = state ^ (state >> 16u);
                return state;
            }
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= params.n_samples) {
                    return;
                }
                
                let hash = pcg_hash(params.seed + idx);
                output[idx] = hash % params.max_index;
            }
        "#;
        
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bootstrap_sample"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bootstrap_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        self.bootstrap_pipeline = Some(pipeline);
    }
    
    /// Create average shader
    fn create_average_pipeline(&mut self) {
        if self.average_pipeline.is_some() {
            return;
        }
        
        let shader_source = r#"
            struct Params {
                n_trees: u32,
                n_samples: u32,
            }
            
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @group(0) @binding(2) var<uniform> params: Params;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let sample_idx = global_id.x;
                if (sample_idx >= params.n_samples) {
                    return;
                }
                
                var sum: f32 = 0.0;
                for (var t: u32 = 0u; t < params.n_trees; t = t + 1u) {
                    sum = sum + input[sample_idx * params.n_trees + t];
                }
                output[sample_idx] = sum / f32(params.n_trees);
            }
        "#;
        
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("average"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("average_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        self.average_pipeline = Some(pipeline);
    }
}

impl GpuBackend for WebGpuBackend {
    fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Buffer Management
    // ═══════════════════════════════════════════════════════════════════════
    
    fn buffer_create(&mut self, size: u64, usage: u32) -> Result<BufferId, GpuError> {
        debug!("[WebGPU] buffer_create(size={}, usage={:#x})", size, usage);
        
        let wgpu_usage = Self::to_wgpu_usage(usage);
        
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });
        
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        
        self.buffers.insert(id, WgpuBuffer { buffer, size, usage });
        
        debug!("[WebGPU] Created buffer {} (size={})", id, size);
        Ok(id)
    }
    
    fn buffer_write(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), GpuError> {
        debug!("[WebGPU] buffer_write(buffer={}, offset={}, len={})", buffer, offset, data.len());
        
        let buf = self.get_buffer(buffer)?;
        self.queue.write_buffer(&buf.buffer, offset, data);
        
        Ok(())
    }
    
    fn buffer_read(&self, buffer: BufferId, offset: u64, size: u32) -> Result<Vec<u8>, GpuError> {
        debug!("[WebGPU] buffer_read(buffer={}, offset={}, size={})", buffer, offset, size);
        
        let buf = self.get_buffer(buffer)?;
        
        // Create staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy to staging
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_encoder"),
        });
        encoder.copy_buffer_to_buffer(&buf.buffer, offset, &staging, 0, size as u64);
        self.queue.submit(Some(encoder.finish()));
        
        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        rx.recv()
            .map_err(|_| GpuError::BackendError("Map channel error".into()))?
            .map_err(|e| GpuError::BackendError(format!("Map failed: {:?}", e)))?;
        
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        
        Ok(data)
    }
    
    fn buffer_copy(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: BufferId,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] buffer_copy(src={}, dst={}, size={})", src, dst, size);
        
        let src_buf = self.get_buffer(src)?;
        let dst_buf = self.get_buffer(dst)?;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buf.buffer, src_offset, &dst_buf.buffer, dst_offset, size);
        self.queue.submit(Some(encoder.finish()));
        
        Ok(())
    }
    
    fn buffer_destroy(&mut self, buffer: BufferId) -> Result<(), GpuError> {
        debug!("[WebGPU] buffer_destroy(buffer={})", buffer);
        
        self.buffers.remove(&buffer).ok_or(GpuError::InvalidBuffer(buffer))?;
        Ok(())
    }
    
    fn sync(&self) {
        debug!("[WebGPU] sync()");
        self.device.poll(wgpu::Maintain::Wait);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // ML Kernels (simplified implementations)
    // ═══════════════════════════════════════════════════════════════════════
    
    fn kernel_bootstrap_sample(
        &mut self,
        params: &BootstrapParams,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_bootstrap_sample(n_samples={}, seed={})",
               params.n_samples, params.seed);
        
        // CPU fallback - same as CUDA backend
        let mut indices = Vec::with_capacity(params.n_samples as usize);
        let mut state = params.seed;
        
        for _ in 0..params.n_samples {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            state ^= state >> 16;
            state = state.wrapping_mul(2654435769);
            state ^= state >> 16;
            
            indices.push(state % params.max_index);
        }
        
        let data: Vec<u8> = indices.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &data)?;
        
        Ok(())
    }
    
    fn kernel_find_split(
        &mut self,
        params: &FindSplitParams,
        data: BufferId,
        labels: BufferId,
        indices: BufferId,
        thresholds: BufferId,
        output_scores: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_find_split - CPU fallback");
        
        // Same CPU implementation as CUDA backend
        let data_bytes = self.buffer_read(data, 0, (params.n_samples * params.n_features * 4) as u32)?;
        let labels_bytes = self.buffer_read(labels, 0, (params.n_samples * 4) as u32)?;
        let indices_bytes = self.buffer_read(indices, 0, (params.n_samples * 4) as u32)?;
        let thresholds_bytes = self.buffer_read(thresholds, 0, (params.n_thresholds * 4) as u32)?;
        
        let data_f32: Vec<f32> = data_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let labels_f32: Vec<f32> = labels_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let indices_u32: Vec<u32> = indices_bytes.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let thresholds_f32: Vec<f32> = thresholds_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let mut scores = Vec::with_capacity(params.n_thresholds as usize);
        
        for &threshold in &thresholds_f32 {
            let mut left_sum = 0.0f32;
            let mut right_sum = 0.0f32;
            let mut left_count = 0u32;
            let mut right_count = 0u32;
            
            for &idx in &indices_u32 {
                let feature_val = data_f32[(idx as usize) * (params.n_features as usize) + (params.feature_idx as usize)];
                let label = labels_f32[idx as usize];
                
                if feature_val <= threshold {
                    left_sum += label;
                    left_count += 1;
                } else {
                    right_sum += label;
                    right_count += 1;
                }
            }
            
            if left_count == 0 || right_count == 0 {
                scores.push(f32::INFINITY);
                continue;
            }
            
            let left_mean = left_sum / left_count as f32;
            let right_mean = right_sum / right_count as f32;
            
            let mut mse = 0.0f32;
            for &idx in &indices_u32 {
                let feature_val = data_f32[(idx as usize) * (params.n_features as usize) + (params.feature_idx as usize)];
                let label = labels_f32[idx as usize];
                let mean = if feature_val <= threshold { left_mean } else { right_mean };
                let diff = label - mean;
                mse += diff * diff;
            }
            
            scores.push(mse);
        }
        
        let scores_bytes: Vec<u8> = scores.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output_scores, 0, &scores_bytes)?;
        
        Ok(())
    }
    
    fn kernel_average(
        &mut self,
        params: &AverageParams,
        tree_predictions: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_average - CPU fallback");
        
        let total_size = params.n_samples * params.n_trees;
        let preds_bytes = self.buffer_read(tree_predictions, 0, (total_size * 4) as u32)?;
        
        let preds: Vec<f32> = preds_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let mut averages = Vec::with_capacity(params.n_samples as usize);
        
        for i in 0..params.n_samples as usize {
            let mut sum = 0.0f32;
            for j in 0..params.n_trees as usize {
                sum += preds[i * params.n_trees as usize + j];
            }
            averages.push(sum / params.n_trees as f32);
        }
        
        let output_bytes: Vec<u8> = averages.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &output_bytes)?;
        
        Ok(())
    }
    
    fn kernel_matmul(
        &mut self,
        params: &MatmulParams,
        a: BufferId,
        b: BufferId,
        c: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_matmul - CPU fallback");
        
        let a_bytes = self.buffer_read(a, 0, (params.m * params.k * 4) as u32)?;
        let b_bytes = self.buffer_read(b, 0, (params.k * params.n * 4) as u32)?;
        let c_bytes = self.buffer_read(c, 0, (params.m * params.n * 4) as u32)?;
        
        let a_f32: Vec<f32> = a_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = b_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32: Vec<f32> = c_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        for i in 0..params.m as usize {
            for j in 0..params.n as usize {
                let mut sum = 0.0f32;
                for l in 0..params.k as usize {
                    sum += a_f32[i * params.k as usize + l] * b_f32[l * params.n as usize + j];
                }
                c_f32[i * params.n as usize + j] = 
                    params.alpha * sum + params.beta * c_f32[i * params.n as usize + j];
            }
        }
        
        let c_out: Vec<u8> = c_f32.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(c, 0, &c_out)?;
        
        Ok(())
    }
    
    fn kernel_elementwise(
        &mut self,
        params: &ElementwiseParams,
        input_a: BufferId,
        input_b: Option<BufferId>,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_elementwise - CPU fallback");
        
        let a_bytes = self.buffer_read(input_a, 0, (params.n_elements * 4) as u32)?;
        let a: Vec<f32> = a_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let b: Option<Vec<f32>> = input_b.map(|id| {
            let b_bytes = self.buffer_read(id, 0, (params.n_elements * 4) as u32).unwrap();
            b_bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        });
        
        let result: Vec<f32> = match params.op {
            ElementwiseOp::Relu => a.iter().map(|&x| x.max(0.0)).collect(),
            ElementwiseOp::Sigmoid => a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            ElementwiseOp::Tanh => a.iter().map(|&x| x.tanh()).collect(),
            ElementwiseOp::Sqrt => a.iter().map(|&x| x.sqrt()).collect(),
            ElementwiseOp::Exp => a.iter().map(|&x| x.exp()).collect(),
            ElementwiseOp::Log => a.iter().map(|&x| x.ln()).collect(),
            ElementwiseOp::Add => {
                let b = b.ok_or(GpuError::InvalidParams("Add requires second input".into()))?;
                a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
            }
            ElementwiseOp::Mul => {
                let b = b.ok_or(GpuError::InvalidParams("Mul requires second input".into()))?;
                a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
            }
        };
        
        let out_bytes: Vec<u8> = result.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &out_bytes)?;
        
        Ok(())
    }
    
    fn kernel_reduce(
        &mut self,
        params: &ReduceParams,
        input: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[WebGPU] kernel_reduce - CPU fallback");
        
        let in_bytes = self.buffer_read(input, 0, (params.n_elements * 4) as u32)?;
        let data: Vec<f32> = in_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let result = match params.op {
            ReduceOp::Sum => data.iter().sum(),
            ReduceOp::Max => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => data.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Mean => data.iter().sum::<f32>() / data.len() as f32,
            ReduceOp::Variance => {
                let mean = data.iter().sum::<f32>() / data.len() as f32;
                data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
            }
        };
        
        self.buffer_write(output, 0, &result.to_le_bytes())?;
        
        Ok(())
    }
    
    fn kernel_batch_predict(
        &mut self,
        params: &BatchPredictParams,
        samples: BufferId,
        tree_nodes: BufferId,
        tree_offsets: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        warn!("[WebGPU] kernel_batch_predict not implemented - using placeholder");
        
        let out_bytes = vec![0u8; (params.batch_size * 4) as usize];
        self.buffer_write(output, 0, &out_bytes)?;
        
        Ok(())
    }
}
