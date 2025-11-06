//! GPU compute acceleration using WebGPU
//! 
//! This module handles parallel prediction across multiple decision trees
//! using WebGPU compute shaders.

use wgpu::{
    Adapter, Device, Queue, Instance, ComputePipeline, Buffer, BindGroup,
    BindGroupLayout, PipelineLayout, ShaderModule,
};
use bytemuck::{Pod, Zeroable};

use crate::random_forest::RandomForest;

/// GPU executor for RandomForest predictions
pub struct GpuExecutor {
    device: Device,
    queue: Queue,
    pipeline: Option<ComputePipeline>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuParams {
    n_trees: u32,
    n_samples: u32,
    padding: [u32; 2], // Align to 16 bytes
}

impl GpuExecutor {
    /// Initialize GPU device and queue
    pub async fn new() -> Result<Self, String> {
        // Create instance
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
            .ok_or("Failed to find an appropriate adapter")?;
        
        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ML Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;
        
        Ok(Self {
            device,
            queue,
            pipeline: None,
        })
    }
    
    /// Run predictions on GPU
    pub async fn predict(
        &self,
        forest: &RandomForest,
        input_data: &[f32],
        n_features: usize,
    ) -> Result<Vec<f32>, String> {
        let n_samples = input_data.len() / n_features;
        let n_trees = forest.n_trees();
        
        // For now, we'll do a simple GPU-accelerated averaging
        // In a full implementation, we'd traverse trees on GPU
        
        // Create buffers
        let mut tree_predictions = Vec::with_capacity(n_samples * n_trees);
        
        // Get predictions from each tree for each sample
        for sample_idx in 0..n_samples {
            let start = sample_idx * n_features;
            let end = start + n_features;
            let sample = &input_data[start..end];
            
            let preds = forest.get_tree_predictions(sample);
            tree_predictions.extend(preds);
        }
        
        // Now use GPU to average the predictions
        let result = self.average_on_gpu(&tree_predictions, n_samples, n_trees).await?;
        
        Ok(result)
    }
    
    /// GPU kernel for averaging tree predictions
    async fn average_on_gpu(
        &self,
        tree_predictions: &[f32],
        n_samples: usize,
        n_trees: usize,
    ) -> Result<Vec<f32>, String> {
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Average Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/average.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Average Bind Group Layout"),
            entries: &[
                // Input buffer (tree predictions)
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
                // Output buffer (averaged predictions)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                    binding: 2,
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
        
        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Average Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Average Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create input buffer
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(tree_predictions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (n_samples * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create params buffer
        let params = GpuParams {
            n_trees: n_trees as u32,
            n_samples: n_samples as u32,
            padding: [0, 0],
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Average Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n_samples * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Average Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch work groups (64 threads per group)
            let workgroups = (n_samples as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (n_samples * std::mem::size_of::<f32>()) as u64,
        );
        
        // Submit commands
        self.queue.submit(Some(encoder.finish()));
        
        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        receiver.receive().await
            .ok_or("Failed to receive buffer mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
}
