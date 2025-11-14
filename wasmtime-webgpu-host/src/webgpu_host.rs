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
        
        // ═══════════════════════════════════════════════════════════════
        // Basic Setup Functions
        // ═══════════════════════════════════════════════════════════════
        
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
        
        // ═══════════════════════════════════════════════════════════════
        // Buffer Operations
        // ═══════════════════════════════════════════════════════════════
        
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
                  data_len: u32| {
                info!("[wasi:webgpu] queue-write-buffer (buffer={}, offset={}, len={})", 
                      buffer_id, offset, data_len);
                      
                let buffers = buffers.lock().unwrap();
                let buffer = match buffers.get(&buffer_id) {
                    Some(b) => b,
                    None => {
                        info!("  ERROR: Invalid buffer ID");
                        return;
                    }
                };
                
                let memory = match caller.get_export("memory") {
                    Some(m) => m.into_memory().expect("Export is not memory"),
                    None => {
                        info!("  ERROR: No memory export");
                        return;
                    }
                };
                
                let data = match memory.data(&caller)
                    .get(data_ptr as usize..(data_ptr + data_len) as usize) {
                    Some(d) => d,
                    None => {
                        info!("  ERROR: Invalid memory access");
                        return;
                    }
                };
                
                gpu.write_buffer(buffer, offset, data);
                info!("  Buffer written successfully");
            },
        )?;
        
        // ═══════════════════════════════════════════════════════════════
        // Shader and Pipeline Operations
        // ═══════════════════════════════════════════════════════════════
        
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
                  code_len: u32| -> u32 {
                info!("[wasi:webgpu] create-shader-module (len={})", code_len);
                
                let memory = match caller.get_export("memory") {
                    Some(m) => m.into_memory().expect("Export is not memory"),
                    None => {
                        info!("  ERROR: No memory export");
                        return 0;
                    }
                };
                
                let code_bytes = match memory.data(&caller)
                    .get(code_ptr as usize..(code_ptr + code_len) as usize) {
                    Some(d) => d,
                    None => {
                        info!("  ERROR: Invalid memory access");
                        return 0;
                    }
                };
                
                let code = match std::str::from_utf8(code_bytes) {
                    Ok(s) => s,
                    Err(_) => {
                        info!("  ERROR: Invalid UTF-8 in shader code");
                        return 0;
                    }
                };
                
                let shader_module = gpu.create_shader_module(code);
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                shader_modules.lock().unwrap().insert(id, shader_module);
                info!("  Created shader module with ID: {}", id);
                id
            },
        )?;
        
        // Create bind group layout
        let bind_group_layouts = self.bind_group_layouts.clone();
        let gpu = self.gpu.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-bind-group-layout",
            move |_caller: Caller<'_, HostState>, 
                  _device: u32,
                  entry_count: u32,
                  entries_ptr: u32| -> u32 {
                info!("[wasi:webgpu] create-bind-group-layout (entries={})", entry_count);
                
                // For now, create a simple layout with storage buffers
                let entries: Vec<BindGroupLayoutEntry> = (0..entry_count)
                    .map(|i| BindGroupLayoutEntry {
                        binding: i,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                
                let layout = gpu.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("WASM Bind Group Layout"),
                    entries: &entries,
                });
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                bind_group_layouts.lock().unwrap().insert(id, layout);
                info!("  Created bind group layout with ID: {}", id);
                id
            },
        )?;
        
        // Create compute pipeline
        let compute_pipelines = self.compute_pipelines.clone();
        let bind_group_layouts = self.bind_group_layouts.clone();
        let shader_modules = self.shader_modules.clone();
        let gpu = self.gpu.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-compute-pipeline",
            move |mut caller: Caller<'_, HostState>,
                  _device: u32,
                  shader_id: u32,
                  entry_point_ptr: u32,
                  entry_point_len: u32,
                  layout_id: u32| -> u32 {
                info!("[wasi:webgpu] create-compute-pipeline (shader={}, layout={})", 
                      shader_id, layout_id);
                
                // Read entry point name
                let memory = match caller.get_export("memory") {
                    Some(m) => m.into_memory().expect("Export is not memory"),
                    None => {
                        info!("  ERROR: No memory export");
                        return 0;
                    }
                };
                
                let entry_bytes = match memory.data(&caller)
                    .get(entry_point_ptr as usize..(entry_point_ptr + entry_point_len) as usize) {
                    Some(d) => d,
                    None => {
                        info!("  ERROR: Invalid memory access");
                        return 0;
                    }
                };
                
                let entry_point = match std::str::from_utf8(entry_bytes) {
                    Ok(s) => s,
                    Err(_) => {
                        info!("  ERROR: Invalid UTF-8 in entry point");
                        return 0;
                    }
                };
                
                // Get shader module
                let shader_modules = shader_modules.lock().unwrap();
                let shader = match shader_modules.get(&shader_id) {
                    Some(s) => s,
                    None => {
                        info!("  ERROR: Invalid shader module ID");
                        return 0;
                    }
                };
                
                // Get bind group layout
                let layouts = bind_group_layouts.lock().unwrap();
                let layout = match layouts.get(&layout_id) {
                    Some(l) => l,
                    None => {
                        info!("  ERROR: Invalid bind group layout ID");
                        return 0;
                    }
                };
                
                // Create pipeline layout
                let pipeline_layout = gpu.device().create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("WASM Pipeline Layout"),
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                });
                
                // Create compute pipeline
                let pipeline = gpu.device().create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("WASM Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: shader,
                    entry_point,
                });
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                compute_pipelines.lock().unwrap().insert(id, pipeline);
                info!("  Created compute pipeline with ID: {}", id);
                id
            },
        )?;
        
        // Create bind group
        let bind_groups = self.bind_groups.clone();
        let bind_group_layouts = self.bind_group_layouts.clone();
        let buffers = self.buffers.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-bind-group",
            move |_caller: Caller<'_, HostState>,
                  _device: u32,
                  layout_id: u32,
                  buffer_count: u32,
                  buffer_ids_ptr: u32| -> u32 {
                info!("[wasi:webgpu] create-bind-group (layout={}, buffers={})", 
                      layout_id, buffer_count);
                
                // Get layout
                let layouts = bind_group_layouts.lock().unwrap();
                let layout = match layouts.get(&layout_id) {
                    Some(l) => l,
                    None => {
                        info!("  ERROR: Invalid bind group layout ID");
                        return 0;
                    }
                };
                
                // For simplicity, assume buffer IDs are sequential starting from buffer_ids_ptr
                // In real implementation, would read from WASM memory
                let buffers_map = buffers.lock().unwrap();
                let mut entries = Vec::new();
                
                for i in 0..buffer_count {
                    let buffer_id = buffer_ids_ptr + i;
                    if let Some(buffer) = buffers_map.get(&buffer_id) {
                        entries.push(BindGroupEntry {
                            binding: i,
                            resource: buffer.as_entire_binding(),
                        });
                    }
                }
                
                let bind_group = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Some("WASM Bind Group"),
                    layout,
                    entries: &entries,
                });
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                bind_groups.lock().unwrap().insert(id, bind_group);
                info!("  Created bind group with ID: {}", id);
                id
            },
        )?;
        
        // ═══════════════════════════════════════════════════════════════
        // Command Encoding and Execution
        // ═══════════════════════════════════════════════════════════════
        
        // Create command encoder
        let command_encoders = self.command_encoders.clone();
        let gpu = self.gpu.clone();
        let next_id = self.next_id.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "create-command-encoder",
            move |_caller: Caller<'_, HostState>, _device: u32| -> u32 {
                info!("[wasi:webgpu] create-command-encoder");
                
                let encoder = gpu.device().create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("WASM Command Encoder"),
                });
                
                let id = {
                    let mut next = next_id.lock().unwrap();
                    let current = *next;
                    *next += 1;
                    current
                };
                
                command_encoders.lock().unwrap().insert(id, encoder);
                info!("  Created command encoder with ID: {}", id);
                id
            },
        )?;
        
        // Begin compute pass and dispatch (combined for simplicity)
        let command_encoders = self.command_encoders.clone();
        let compute_pipelines = self.compute_pipelines.clone();
        let bind_groups = self.bind_groups.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "dispatch-compute",
            move |_caller: Caller<'_, HostState>,
                  encoder_id: u32,
                  pipeline_id: u32,
                  bind_group_id: u32,
                  workgroup_count_x: u32,
                  workgroup_count_y: u32,
                  workgroup_count_z: u32| {
                info!("[wasi:webgpu] dispatch-compute ({}x{}x{})", 
                      workgroup_count_x, workgroup_count_y, workgroup_count_z);
                
                let mut encoders = command_encoders.lock().unwrap();
                let encoder = match encoders.get_mut(&encoder_id) {
                    Some(e) => e,
                    None => {
                        info!("  ERROR: Invalid encoder ID");
                        return;
                    }
                };
                
                let pipelines = compute_pipelines.lock().unwrap();
                let pipeline = match pipelines.get(&pipeline_id) {
                    Some(p) => p,
                    None => {
                        info!("  ERROR: Invalid pipeline ID");
                        return;
                    }
                };
                
                let bind_groups_map = bind_groups.lock().unwrap();
                let bind_group = match bind_groups_map.get(&bind_group_id) {
                    Some(bg) => bg,
                    None => {
                        info!("  ERROR: Invalid bind group ID");
                        return;
                    }
                };
                
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("WASM Compute Pass"),
                    timestamp_writes: None,
                });
                
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z);
                
                drop(compute_pass);
                info!("  Compute pass dispatched");
            },
        )?;
        
        // Submit command buffer
        let command_encoders = self.command_encoders.clone();
        let gpu = self.gpu.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "submit-commands",
            move |_caller: Caller<'_, HostState>, _queue: u32, encoder_id: u32| {
                info!("[wasi:webgpu] submit-commands (encoder={})", encoder_id);
                
                let mut encoders = command_encoders.lock().unwrap();
                if let Some(encoder) = encoders.remove(&encoder_id) {
                    let command_buffer = encoder.finish();
                    gpu.submit(command_buffer);
                    info!("  Commands submitted");
                } else {
                    info!("  ERROR: Invalid encoder ID");
                }
            },
        )?;
        
        // Buffer map async (simplified - just polls device)
        let gpu = self.gpu.clone();
        let buffers = self.buffers.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "buffer-map-async",
            move |mut caller: Caller<'_, HostState>,
                  buffer_id: u32,
                  mode: u32,
                  offset: u64,
                  size: u64,
                  callback_ptr: u32,
                  callback_len: u32| {
                info!("[wasi:webgpu] buffer-map-async (buffer={}, mode={}, size={})", 
                      buffer_id, mode, size);
                
                let buffers_map = buffers.lock().unwrap();
                if let Some(buffer) = buffers_map.get(&buffer_id) {
                    let buffer_slice = buffer.slice(offset..offset + size);
                    
                    let map_mode = if mode == 1 {
                        MapMode::Read
                    } else {
                        MapMode::Write
                    };
                    
                    buffer_slice.map_async(map_mode, move |result| {
                        info!("  Buffer mapped: {:?}", result);
                    });
                    
                    // Poll device to complete the mapping
                    gpu.device().poll(Maintain::Wait);
                    info!("  Buffer mapping complete");
                } else {
                    info!("  ERROR: Invalid buffer ID");
                }
            },
        )?;
        
        // Get mapped range (copy data back to WASM memory)
        let buffers = self.buffers.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "buffer-get-mapped-range",
            move |mut caller: Caller<'_, HostState>,
                  buffer_id: u32,
                  offset: u64,
                  size: u64,
                  dest_ptr: u32| {
                info!("[wasi:webgpu] buffer-get-mapped-range (buffer={}, size={})", 
                      buffer_id, size);
                
                let buffers_map = buffers.lock().unwrap();
                if let Some(buffer) = buffers_map.get(&buffer_id) {
                    let buffer_slice = buffer.slice(offset..offset + size);
                    let data = buffer_slice.get_mapped_range();
                    
                    let memory = match caller.get_export("memory") {
                        Some(m) => m.into_memory().expect("Export is not memory"),
                        None => {
                            info!("  ERROR: No memory export");
                            return;
                        }
                    };
                    
                    if let Some(dest) = memory.data_mut(&mut caller)
                        .get_mut(dest_ptr as usize..(dest_ptr as usize + size as usize)) {
                        dest.copy_from_slice(&data);
                        info!("  Copied {} bytes to WASM memory", size);
                    } else {
                        info!("  ERROR: Invalid destination memory");
                    }
                } else {
                    info!("  ERROR: Invalid buffer ID");
                }
            },
        )?;
        
        // Buffer unmap
        let buffers = self.buffers.clone();
        
        linker.func_wrap(
            "wasi:webgpu",
            "buffer-unmap",
            move |_caller: Caller<'_, HostState>, buffer_id: u32| {
                info!("[wasi:webgpu] buffer-unmap (buffer={})", buffer_id);
                
                let buffers_map = buffers.lock().unwrap();
                if let Some(buffer) = buffers_map.get(&buffer_id) {
                    buffer.unmap();
                    info!("  Buffer unmapped");
                } else {
                    info!("  ERROR: Invalid buffer ID");
                }
            },
        )?;
        
        info!("{} wasi:webgpu functions registered", 17);
        
        Ok(())
    }
}
