//! WASI GPU Host Functions
//!
//! Implements the wasi:gpu interface as Wasmtime host functions.
//! These bridge between the WASM module and the actual GPU backend.

use crate::backend::{
    AverageParams, BatchPredictParams, BootstrapParams, BufferId, DeviceInfo,
    ElementwiseOp, ElementwiseParams, FindSplitParams, GpuBackend, GpuError,
    MatmulParams, ReduceOp, ReduceParams,
};
use tracing::{debug, error, info};
use wasmtime::{Caller, Linker};

/// GPU State accessible from host functions
pub struct GpuState {
    backend: Box<dyn GpuBackend>,
}

impl GpuState {
    pub fn new(backend: Box<dyn GpuBackend>) -> Self {
        Self { backend }
    }
    
    pub fn backend(&self) -> &dyn GpuBackend {
        self.backend.as_ref()
    }
    
    pub fn backend_mut(&mut self) -> &mut dyn GpuBackend {
        self.backend.as_mut()
    }
}

/// Error code conversion
fn error_to_code(err: &GpuError) -> u32 {
    match err {
        GpuError::OutOfMemory => 1,
        GpuError::InvalidBuffer(_) => 2,
        GpuError::KernelError(_) => 3,
        GpuError::DeviceUnavailable => 4,
        GpuError::InvalidParams(_) => 5,
        GpuError::BackendError(_) => 6,
    }
}

/// Helper to read bytes from WASM memory
fn read_memory<T>(caller: &Caller<'_, T>, ptr: u32, len: usize) -> Option<Vec<u8>> {
    let memory = caller.get_export("memory")?.into_memory()?;
    let mut buf = vec![0u8; len];
    memory.read(caller, ptr as usize, &mut buf).ok()?;
    Some(buf)
}

/// Helper to write bytes to WASM memory
fn write_memory<T>(caller: &mut Caller<'_, T>, ptr: u32, data: &[u8]) -> bool {
    if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
        memory.write(caller, ptr as usize, data).is_ok()
    } else {
        false
    }
}

/// Register all wasi:gpu host functions with the linker
pub fn add_to_linker<T>(
    linker: &mut Linker<T>,
    get_state: impl Fn(&mut T) -> &mut GpuState + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/compute interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // get-device-info
    linker.func_wrap(
        "wasi:gpu/compute",
        "get-device-info",
        move |mut caller: Caller<'_, T>,
              name_ptr: u32, name_len: u32,
              backend_ptr: u32, backend_len: u32,
              memory_ptr: u32,
              hardware_ptr: u32,
              compute_cap_ptr: u32, compute_cap_len: u32| {
            
            // Get device info first (borrow state)
            let (name, backend, total_memory, is_hardware, compute_cap) = {
                let state = get_state(caller.data_mut());
                let info = state.backend().device_info();
                (
                    info.name.clone(),
                    info.backend.clone(),
                    info.total_memory,
                    info.is_hardware,
                    info.compute_capability.clone(),
                )
            };
            
            // Now write to memory (state borrow released)
            let name_bytes = name.as_bytes();
            let name_write_len = name_bytes.len().min(name_len as usize);
            write_memory(&mut caller, name_ptr, &name_bytes[..name_write_len]);
            
            let backend_bytes = backend.as_bytes();
            let backend_write_len = backend_bytes.len().min(backend_len as usize);
            write_memory(&mut caller, backend_ptr, &backend_bytes[..backend_write_len]);
            
            write_memory(&mut caller, memory_ptr, &total_memory.to_le_bytes());
            
            let hw: u32 = if is_hardware { 1 } else { 0 };
            write_memory(&mut caller, hardware_ptr, &hw.to_le_bytes());
            
            let cc_bytes = compute_cap.as_bytes();
            let cc_write_len = cc_bytes.len().min(compute_cap_len as usize);
            write_memory(&mut caller, compute_cap_ptr, &cc_bytes[..cc_write_len]);
        },
    )?;
    
    // buffer-create
    linker.func_wrap(
        "wasi:gpu/compute",
        "buffer-create",
        move |mut caller: Caller<'_, T>, size: u64, usage: u32, out_ptr: u32| -> u32 {
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_create(size, usage) {
                Ok(id) => {
                    write_memory(&mut caller, out_ptr, &id.to_le_bytes());
                    0 // Success
                }
                Err(e) => {
                    error!("[Host] buffer-create failed: {:?}", e);
                    error_to_code(&e)
                }
            }
        },
    )?;
    
    // buffer-write
    linker.func_wrap(
        "wasi:gpu/compute",
        "buffer-write",
        move |mut caller: Caller<'_, T>, buffer: u32, offset: u64, data_ptr: u32, data_len: u32| -> u32 {
            // Read data from WASM memory first
            let data = match read_memory(&caller, data_ptr, data_len as usize) {
                Some(d) => d,
                None => return 5, // Invalid params
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_write(buffer, offset, &data) {
                Ok(()) => 0,
                Err(e) => {
                    error!("[Host] buffer-write failed: {:?}", e);
                    error_to_code(&e)
                }
            }
        },
    )?;
    
    // buffer-read
    linker.func_wrap(
        "wasi:gpu/compute",
        "buffer-read",
        move |mut caller: Caller<'_, T>, buffer: u32, offset: u64, size: u32, out_ptr: u32, out_len_ptr: u32| -> u32 {
            // Read from backend first
            let data = {
                let state = get_state(caller.data_mut());
                match state.backend().buffer_read(buffer, offset, size) {
                    Ok(d) => d,
                    Err(e) => {
                        error!("[Host] buffer-read failed: {:?}", e);
                        return error_to_code(&e);
                    }
                }
            };
            
            // Write to WASM memory
            write_memory(&mut caller, out_ptr, &data);
            write_memory(&mut caller, out_len_ptr, &(data.len() as u32).to_le_bytes());
            0
        },
    )?;
    
    // buffer-copy
    linker.func_wrap(
        "wasi:gpu/compute",
        "buffer-copy",
        move |mut caller: Caller<'_, T>, src: u32, src_offset: u64, dst: u32, dst_offset: u64, size: u64| -> u32 {
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_copy(src, src_offset, dst, dst_offset, size) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // buffer-destroy
    linker.func_wrap(
        "wasi:gpu/compute",
        "buffer-destroy",
        move |mut caller: Caller<'_, T>, buffer: u32| -> u32 {
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_destroy(buffer) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // sync
    linker.func_wrap(
        "wasi:gpu/compute",
        "sync",
        move |mut caller: Caller<'_, T>| {
            let state = get_state(caller.data_mut());
            state.backend().sync();
        },
    )?;
    
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/ml-kernels interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // kernel-bootstrap-sample
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-bootstrap-sample",
        move |mut caller: Caller<'_, T>, params_ptr: u32, output: u32| -> u32 {
            // Read params first
            let params_bytes = match read_memory(&caller, params_ptr, 12) {
                Some(b) => b,
                None => return 5,
            };
            
            let params = BootstrapParams {
                n_samples: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                seed: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                max_index: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_bootstrap_sample(&params, output) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-find-split
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-find-split",
        move |mut caller: Caller<'_, T>, 
              params_ptr: u32,
              data: u32, labels: u32, indices: u32, thresholds: u32, output_scores: u32| -> u32 {
            
            let params_bytes = match read_memory(&caller, params_ptr, 16) {
                Some(b) => b,
                None => return 5,
            };
            
            let params = FindSplitParams {
                n_samples: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_features: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                feature_idx: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                n_thresholds: u32::from_le_bytes([params_bytes[12], params_bytes[13], params_bytes[14], params_bytes[15]]),
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_find_split(&params, data, labels, indices, thresholds, output_scores) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-average
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-average",
        move |mut caller: Caller<'_, T>, params_ptr: u32, tree_predictions: u32, output: u32| -> u32 {
            let params_bytes = match read_memory(&caller, params_ptr, 8) {
                Some(b) => b,
                None => return 5,
            };
            
            let params = AverageParams {
                n_trees: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_samples: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_average(&params, tree_predictions, output) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-matmul
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-matmul",
        move |mut caller: Caller<'_, T>, params_ptr: u32, a: u32, b: u32, c: u32| -> u32 {
            let params_bytes = match read_memory(&caller, params_ptr, 24) {
                Some(b) => b,
                None => return 5,
            };
            
            let params = MatmulParams {
                m: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                k: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                n: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                trans_a: params_bytes[12] != 0,
                trans_b: params_bytes[16] != 0,
                alpha: f32::from_le_bytes([params_bytes[20], params_bytes[21], params_bytes[22], params_bytes[23]]),
                beta: 0.0,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_matmul(&params, a, b, c) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-elementwise
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-elementwise",
        move |mut caller: Caller<'_, T>, params_ptr: u32, input_a: u32, input_b: u32, output: u32| -> u32 {
            let params_bytes = match read_memory(&caller, params_ptr, 8) {
                Some(b) => b,
                None => return 5,
            };
            
            let n_elements = u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]);
            let op_code = u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]);
            
            let op = match op_code {
                0 => ElementwiseOp::Relu,
                1 => ElementwiseOp::Sigmoid,
                2 => ElementwiseOp::Tanh,
                3 => ElementwiseOp::Add,
                4 => ElementwiseOp::Mul,
                5 => ElementwiseOp::Sqrt,
                6 => ElementwiseOp::Exp,
                7 => ElementwiseOp::Log,
                _ => return 5,
            };
            
            let params = ElementwiseParams { n_elements, op };
            let input_b_opt = if input_b == 0 { None } else { Some(input_b) };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_elementwise(&params, input_a, input_b_opt, output) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-reduce
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-reduce",
        move |mut caller: Caller<'_, T>, params_ptr: u32, input: u32, output: u32| -> u32 {
            let params_bytes = match read_memory(&caller, params_ptr, 8) {
                Some(b) => b,
                None => return 5,
            };
            
            let n_elements = u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]);
            let op_code = u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]);
            
            let op = match op_code {
                0 => ReduceOp::Sum,
                1 => ReduceOp::Max,
                2 => ReduceOp::Min,
                3 => ReduceOp::Mean,
                4 => ReduceOp::Variance,
                _ => return 5,
            };
            
            let params = ReduceParams { n_elements, op };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_reduce(&params, input, output) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    // kernel-batch-predict
    linker.func_wrap(
        "wasi:gpu/ml-kernels",
        "kernel-batch-predict",
        move |mut caller: Caller<'_, T>, 
              params_ptr: u32,
              samples: u32, tree_nodes: u32, tree_offsets: u32, output: u32| -> u32 {
            
            let params_bytes = match read_memory(&caller, params_ptr, 16) {
                Some(b) => b,
                None => return 5,
            };
            
            let params = BatchPredictParams {
                batch_size: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_features: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                n_trees: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                max_depth: u32::from_le_bytes([params_bytes[12], params_bytes[13], params_bytes[14], params_bytes[15]]),
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_batch_predict(&params, samples, tree_nodes, tree_offsets, output) {
                Ok(()) => 0,
                Err(e) => error_to_code(&e)
            }
        },
    )?;
    
    info!("[Host] wasi:gpu host functions registered");
    
    Ok(())
}
