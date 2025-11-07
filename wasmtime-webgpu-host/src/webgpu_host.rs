/**
 * WebGPU Host Implementation
 * 
 * Implements wasi:webgpu functions that can be called from WASM
 */

use anyhow::Result;
use wasmtime::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::*;
use log::info;

use crate::gpu_backend::GpuBackend;
use crate::HostState;

/// Resource IDs for GPU objects
type ResourceId = u32;

pub struct WebGpuHost {
    gpu: Arc<GpuBackend>,
    buffers: Arc<Mutex<HashMap<ResourceId, Buffer>>>,
    shader_modules: Arc<Mutex<HashMap<ResourceId, ShaderModule>>>,
    compute_pipelines: Arc<Mutex<HashMap<ResourceId, ComputePipeline>>>,
    bind_groups: Arc<Mutex<HashMap<ResourceId, BindGroup>>>,
    bind_group_layouts: Arc<Mutex<HashMap<ResourceId, BindGroupLayout>>>,
    command_encoders: Arc<Mutex<HashMap<ResourceId, CommandEncoder>>>,
    next_id: Arc<Mutex<ResourceId>>,
}

impl WebGpuHost {
    pub fn new(gpu: GpuBackend) -> Self {
        Self {
            gpu: Arc::new(gpu),
            buffers: Arc::new(Mutex::new(HashMap::new())),
            shader_modules: Arc::new(Mutex::new(HashMap::new())),
            compute_pipelines: Arc::new(Mutex::new(HashMap::new())),
            bind_groups: Arc::new(Mutex::new(HashMap::new())),
            bind_group_layouts: Arc::new(Mutex::new(HashMap::new())),
            command_encoders: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
    
    fn alloc_id(&self) -> ResourceId {
        let mut id = self.next_id.lock().unwrap();
        let current = *id;
        *id += 1;
        current
    }
    
    /// Register wasi:webgpu functions with the linker
    pub fn register_functions(&self, linker: &mut Linker<HostState>) -> Result<()> {
        info!("Registering wasi:webgpu functions...");
        
        // Create instance (always returns 1 for singleton)
        linker.func_wrap(
            "wasi:webgpu",
            "create-instance",
            |_caller: Caller<'_, HostState>| -> u32 {
                info!("[wasi:webgpu] create-instance");
                1 // Singleton instance ID
            },
        )?;
        
        // Request adapter
        linker.func_wrap(
            "wasi:webgpu",
            "request-adapter",
            |_caller: Caller<'_, HostState>, _instance: u32, _power_pref: u32| -> u32 {
                info!("[wasi:webgpu] request-adapter");
                1 // Singleton adapter ID
            },
        )?;
        
        // Request device
        linker.func_wrap(
            "wasi:webgpu",
            "request-device",
            |_caller: Caller<'_, HostState>, _adapter: u32| -> u32 {
                info!("[wasi:webgpu] request-device");
                1 // Singleton device ID
            },
        )?;
        
        // Get queue
        linker.func_wrap(
            "wasi:webgpu",
            "get-queue",
            |_caller: Caller<'_, HostState>, _device: u32| -> u32 {
                info!("[wasi:webgpu] get-queue");
                1 // Singleton queue ID
            },
        )?;
        
        // Create buffer
        let buffers = self.buffers.clone();
        let gpu = self.gpu.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-buffer",
            move |_caller: Caller<'_, HostState>, _device: u32, size: u64, usage: u32| -> u32 {
                info!("[wasi:webgpu] create-buffer (size={}, usage={})", size, usage);
                
                // Convert usage flags
                let mut buffer_usage = BufferUsages::empty();
                if usage & 0x0001 != 0 { buffer_usage |= BufferUsages::MAP_READ; }
                if usage & 0x0002 != 0 { buffer_usage |= BufferUsages::MAP_WRITE; }
                if usage & 0x0004 != 0 { buffer_usage |= BufferUsages::COPY_SRC; }
                if usage & 0x0008 != 0 { buffer_usage |= BufferUsages::COPY_DST; }
                if usage & 0x0040 != 0 { buffer_usage |= BufferUsages::UNIFORM; }
                if usage & 0x0080 != 0 { buffer_usage |= BufferUsages::STORAGE; }
                
                let buffer = gpu.create_buffer(size, buffer_usage, false);
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                buffers.lock().unwrap().insert(id, buffer);
                info!("  Created buffer with ID: {}", id);
                id
            },
        )?;
        
        // Queue write buffer
        let buffers = self.buffers.clone();
        let gpu = self.gpu.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "queue-write-buffer",
            move |mut caller: Caller<'_, HostState>, 
                  _queue: u32, 
                  buffer_id: u32, 
                  offset: u64, 
                  data_ptr: u32, 
                  data_len: u32| -> Result<(), Trap> {
                
                info!("[wasi:webgpu] queue-write-buffer (buffer={}, offset={}, len={})", 
                      buffer_id, offset, data_len);
                
                // Get buffer
                let buffers = buffers.lock().unwrap();
                let buffer = buffers.get(&buffer_id)
                    .ok_or_else(|| Trap::new("Invalid buffer ID"))?;
                
                // Read data from WASM memory
                let memory = caller.get_export("memory")
                    .ok_or_else(|| Trap::new("No memory export"))?
                    .into_memory()
                    .ok_or_else(|| Trap::new("Export is not memory"))?;
                
                let data = memory.data(&caller)
                    .get(data_ptr as usize..(data_ptr + data_len) as usize)
                    .ok_or_else(|| Trap::new("Invalid memory access"))?;
                
                gpu.write_buffer(buffer, offset, data);
                
                Ok(())
            },
        )?;
        
        // Create shader module
        let shader_modules = self.shader_modules.clone();
        let gpu = self.gpu.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-shader-module",
            move |mut caller: Caller<'_, HostState>, 
                  _device: u32, 
                  code_ptr: u32, 
                  code_len: u32| -> Result<u32, Trap> {
                
                info!("[wasi:webgpu] create-shader-module");
                
                // Read shader code from WASM memory
                let memory = caller.get_export("memory")
                    .ok_or_else(|| Trap::new("No memory export"))?
                    .into_memory()
                    .ok_or_else(|| Trap::new("Export is not memory"))?;
                
                let code_bytes = memory.data(&caller)
                    .get(code_ptr as usize..(code_ptr + code_len) as usize)
                    .ok_or_else(|| Trap::new("Invalid memory access"))?;
                
                let code = std::str::from_utf8(code_bytes)
                    .map_err(|_| Trap::new("Invalid UTF-8 in shader code"))?;
                
                let shader_module = gpu.create_shader_module(code);
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                shader_modules.lock().unwrap().insert(id, shader_module);
                info!("  Created shader module with ID: {}", id);
                
                Ok(id)
            },
        )?;
        
        info!("✓ {} wasi:webgpu functions registered", 7);
        
        Ok(())
    }
}
