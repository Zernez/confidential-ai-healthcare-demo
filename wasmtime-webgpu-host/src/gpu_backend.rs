/**
 * GPU Backend using wgpu
 * 
 * Provides actual GPU access through wgpu library
 */

use anyhow::Result;
use wgpu::*;
use std::sync::Arc;
use log::info;

pub struct GpuBackend {
    instance: Arc<Instance>,
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl GpuBackend {
    pub async fn new() -> Result<Self> {
        info!("Creating GPU instance...");
        
        // Create instance
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });
        
        // Request adapter
        info!("Requesting GPU adapter...");
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable GPU adapter"))?;
        
        let adapter_info = adapter.get_info();
        info!("GPU Adapter found:");
        info!("  Name: {}", adapter_info.name);
        info!("  Backend: {:?}", adapter_info.backend);
        info!("  Device Type: {:?}", adapter_info.device_type);
        
        // Request device and queue
        info!("Requesting GPU device...");
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("WebGPU Host Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await?;
        
        info!("✓ GPU device created");
        
        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
    
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    
    /// Create a buffer
    pub fn create_buffer(&self, size: u64, usage: BufferUsages, mapped_at_creation: bool) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some("WebGPU Host Buffer"),
            size,
            usage,
            mapped_at_creation,
        })
    }
    
    /// Create shader module from WGSL
    pub fn create_shader_module(&self, code: &str) -> ShaderModule {
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("WebGPU Host Shader"),
            source: ShaderSource::Wgsl(code.into()),
        })
    }
    
    /// Write to buffer
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }
    
    /// Read from buffer (async)
    pub async fn read_buffer(&self, buffer: &Buffer, size: u64) -> Result<Vec<u8>> {
        let staging_buffer = self.create_buffer(
            size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            false,
        );
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Buffer Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        
        self.queue.submit(Some(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        self.device.poll(Maintain::Wait);
        
        receiver.await??;
        
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
    
    /// Submit command buffer
    pub fn submit(&self, command_buffer: CommandBuffer) {
        self.queue.submit(Some(command_buffer));
    }
}
