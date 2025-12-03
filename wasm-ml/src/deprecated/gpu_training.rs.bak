//! GPU-accelerated training for RandomForest
//! 
//! This module implements GPU compute shaders for:
//! 1. Bootstrap sampling (random index generation)
//! 2. Best split finding (parallel MSE computation)
//! 3. Histogram building for split candidates

use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup, CommandEncoder};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BootstrapParams {
    n_samples: u32,
    seed: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SplitParams {
    n_samples: u32,
    n_features: u32,
    feature_idx: u32,
    n_thresholds: u32,
}

pub struct GpuTrainer {
    device: Device,
    queue: Queue,
    bootstrap_pipeline: ComputePipeline,
    split_pipeline: ComputePipeline,
}

impl GpuTrainer {
    pub async fn new(device: Device, queue: Queue) -> Result<Self, String> {
        // Create bootstrap sampling pipeline
        let bootstrap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bootstrap Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bootstrap_sample.wgsl").into()),
        });
        
        let bootstrap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bootstrap Layout"),
            entries: &[
                // Output indices
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bootstrap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bootstrap Pipeline Layout"),
            bind_group_layouts: &[&bootstrap_layout],
            push_constant_ranges: &[],
        });
        
        let bootstrap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bootstrap Pipeline"),
            layout: Some(&bootstrap_pipeline_layout),
            module: &bootstrap_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create split finding pipeline
        let split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Split Finding Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/find_split.wgsl").into()),
        });
        
        let split_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Split Layout"),
            entries: &[
                // Input data
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Labels
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bootstrap indices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Thresholds
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output scores
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let split_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Split Pipeline Layout"),
            bind_group_layouts: &[&split_layout],
            push_constant_ranges: &[],
        });
        
        let split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Split Pipeline"),
            layout: Some(&split_pipeline_layout),
            module: &split_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        Ok(Self {
            device,
            queue,
            bootstrap_pipeline,
            split_pipeline,
        })
    }
    
    /// Generate bootstrap sample indices on GPU
    pub async fn bootstrap_sample(&self, n_samples: usize, seed: u32) -> Result<Vec<u32>, String> {
        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bootstrap Output"),
            size: (n_samples * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create params buffer
        let params = BootstrapParams {
            n_samples: n_samples as u32,
            seed,
            padding: [0, 0],
        };
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bootstrap Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bootstrap Bind Group"),
            layout: &self.bootstrap_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create staging buffer for reading
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bootstrap Staging"),
            size: (n_samples * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Execute compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bootstrap Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bootstrap Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.bootstrap_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch: 256 threads per workgroup
            let workgroups = (n_samples as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (n_samples * std::mem::size_of::<u32>()) as u64,
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        receiver.receive().await
            .ok_or("Failed to receive buffer mapping")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let indices: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(indices)
    }
    
    /// Find best split on GPU for a given feature
    pub async fn find_best_split(
        &self,
        data: &[f32],
        labels: &[f32],
        indices: &[u32],
        n_features: usize,
        feature_idx: usize,
    ) -> Result<(f32, f32), String> {
        let n_samples = indices.len();
        
        // Extract feature values and compute candidate thresholds
        let mut feature_values: Vec<f32> = indices
            .iter()
            .map(|&idx| data[idx as usize * n_features + feature_idx])
            .collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_values.dedup();
        
        if feature_values.len() < 2 {
            return Ok((0.0, f32::INFINITY));
        }
        
        // Compute thresholds (midpoints)
        let thresholds: Vec<f32> = feature_values
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();
        
        let n_thresholds = thresholds.len();
        
        // Create GPU buffers
        let data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let labels_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Labels Buffer"),
            contents: bytemuck::cast_slice(labels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let indices_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let thresholds_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Thresholds Buffer"),
            contents: bytemuck::cast_slice(&thresholds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let scores_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Buffer"),
            size: (n_thresholds * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let params = SplitParams {
            n_samples: n_samples as u32,
            n_features: n_features as u32,
            feature_idx: feature_idx as u32,
            n_thresholds: n_thresholds as u32,
        };
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Split Bind Group"),
            layout: &self.split_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: labels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: thresholds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Staging"),
            size: (n_thresholds * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Split Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Split Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.split_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // 64 threads per workgroup
            let workgroups = (n_thresholds as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        encoder.copy_buffer_to_buffer(
            &scores_buffer,
            0,
            &staging_buffer,
            0,
            (n_thresholds * std::mem::size_of::<f32>()) as u64,
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // Read scores
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        receiver.receive().await
            .ok_or("Failed to receive buffer mapping")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        // Find best threshold
        let (best_idx, &best_score) = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or("No valid split found")?;
        
        let best_threshold = thresholds[best_idx];
        
        Ok((best_threshold, best_score))
    }
}
