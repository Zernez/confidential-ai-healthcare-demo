//! WASI GPU Host Functions
//!
//! Implements the wasi:gpu interface as Wasmtime host functions.
//! These bridge between the WASM module and the actual GPU backend.
//!
//! The ABI follows wit-bindgen's conventions for result types:
//! - Functions with result<T, E> take an extra retptr parameter
//! - Return value: 0 = success (value at retptr), 1 = error (error at retptr)

use crate::backend::{
    AverageParams, BatchPredictParams, BootstrapParams,
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

/// Error code for GpuError variant discrimination
fn error_discriminant(err: &GpuError) -> u32 {
    match err {
        GpuError::OutOfMemory => 0,
        GpuError::InvalidBuffer(_) => 1,
        GpuError::KernelError(_) => 2,
        GpuError::DeviceUnavailable => 3,
        GpuError::InvalidParams(_) => 4,
        GpuError::BackendError(_) => 5,
    }
}

/// Helper to read bytes from WASM memory
fn read_memory<T>(caller: &mut Caller<'_, T>, ptr: u32, len: usize) -> Option<Vec<u8>> {
    let memory = caller.get_export("memory")?.into_memory()?;
    let mut buf = vec![0u8; len];
    memory.read(&*caller, ptr as usize, &mut buf).ok()?;
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

/// Write a u32 to WASM memory
fn write_u32<T>(caller: &mut Caller<'_, T>, ptr: u32, value: u32) -> bool {
    write_memory(caller, ptr, &value.to_le_bytes())
}

/// Write a u64 to WASM memory  
fn write_u64<T>(caller: &mut Caller<'_, T>, ptr: u32, value: u64) -> bool {
    write_memory(caller, ptr, &value.to_le_bytes())
}

/// Register all wasi:gpu host functions with the linker
/// 
/// Functions are registered with versioned module names to match WIT component model:
/// - wasi:gpu/compute@0.1.0
/// - wasi:gpu/ml-kernels@0.1.0
pub fn add_to_linker<T: 'static>(
    linker: &mut Linker<T>,
    get_state: impl Fn(&mut T) -> &mut GpuState + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    // Module names for the WIT interface
    const COMPUTE_MODULE: &str = "wasi:gpu/compute@0.1.0";
    const ML_KERNELS_MODULE: &str = "wasi:gpu/ml-kernels@0.1.0";
    
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/compute@0.1.0 interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // get-device-info: func() -> device-info
    // ABI: (retptr: i32) -> ()
    // Writes DeviceInfo struct to retptr
    linker.func_wrap(
        COMPUTE_MODULE,
        "get-device-info",
        move |mut caller: Caller<'_, T>, retptr: i32| {
            debug!("[Host] get-device-info called");
            
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
            
            // Write strings and struct to retptr
            // DeviceInfo layout: name_ptr, name_len, backend_ptr, backend_len, total_memory, is_hardware, cc_ptr, cc_len
            let retptr = retptr as u32;
            
            // For simplicity, write the full struct at retptr
            // Actual layout depends on wit-bindgen version
            let name_bytes = name.as_bytes();
            let backend_bytes = backend.as_bytes();
            let cc_bytes = compute_cap.as_bytes();
            
            // Write name at offset 0 (after struct)
            let name_offset = retptr + 48; // After struct fields
            write_memory(&mut caller, name_offset, name_bytes);
            write_u32(&mut caller, retptr, name_offset);
            write_u32(&mut caller, retptr + 4, name_bytes.len() as u32);
            
            // Write backend at offset after name
            let backend_offset = name_offset + name_bytes.len() as u32;
            write_memory(&mut caller, backend_offset, backend_bytes);
            write_u32(&mut caller, retptr + 8, backend_offset);
            write_u32(&mut caller, retptr + 12, backend_bytes.len() as u32);
            
            // Write total_memory
            write_u64(&mut caller, retptr + 16, total_memory);
            
            // Write is_hardware
            write_u32(&mut caller, retptr + 24, if is_hardware { 1 } else { 0 });
            
            // Write compute_capability
            let cc_offset = backend_offset + backend_bytes.len() as u32;
            write_memory(&mut caller, cc_offset, cc_bytes);
            write_u32(&mut caller, retptr + 28, cc_offset);
            write_u32(&mut caller, retptr + 32, cc_bytes.len() as u32);
        },
    )?;
    
    // buffer-create: func(size: u64, usage: buffer-usage) -> result<buffer-id, gpu-error>
    // ABI: (size: i64, usage: i32, retptr: i32) -> ()
    // Writes: discriminant (0=ok, 1=err) + payload at retptr
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-create",
        move |mut caller: Caller<'_, T>, size: i64, usage: i32, retptr: i32| {
            debug!("[Host] buffer-create(size={}, usage={:#x})", size, usage);
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().buffer_create(size as u64, usage as u32) {
                Ok(id) => {
                    write_u32(&mut caller, retptr, 0); // Ok discriminant
                    write_u32(&mut caller, retptr + 4, id); // buffer-id
                }
                Err(e) => {
                    error!("[Host] buffer-create failed: {:?}", e);
                    write_u32(&mut caller, retptr, 1); // Err discriminant
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-write: func(buffer: buffer-id, offset: u64, data: list<u8>) -> result<_, gpu-error>
    // ABI: (buffer: i32, offset: i64, data_ptr: i32, data_len: i32, retptr: i32) -> ()
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-write",
        move |mut caller: Caller<'_, T>, buffer: i32, offset: i64, data_ptr: i32, data_len: i32, retptr: i32| {
            debug!("[Host] buffer-write(buffer={}, offset={}, len={})", buffer, offset, data_len);
            
            // Read data from WASM memory first
            let data = match read_memory(&mut caller, data_ptr as u32, data_len as usize) {
                Some(d) => d,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4); // invalid-params
                    return;
                }
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().buffer_write(buffer as u32, offset as u64, &data) {
                Ok(()) => {
                    write_u32(&mut caller, retptr, 0); // Ok
                }
                Err(e) => {
                    error!("[Host] buffer-write failed: {:?}", e);
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-read: func(buffer: buffer-id, offset: u64, size: u32) -> result<list<u8>, gpu-error>
    // ABI: (buffer: i32, offset: i64, size: i32, retptr: i32) -> ()
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-read",
        move |mut caller: Caller<'_, T>, buffer: i32, offset: i64, size: i32, retptr: i32| {
            debug!("[Host] buffer-read(buffer={}, offset={}, size={})", buffer, offset, size);
            
            let data = {
                let state = get_state(caller.data_mut());
                match state.backend().buffer_read(buffer as u32, offset as u64, size as u32) {
                    Ok(d) => d,
                    Err(e) => {
                        error!("[Host] buffer-read failed: {:?}", e);
                        write_u32(&mut caller, retptr as u32, 1);
                        write_u32(&mut caller, retptr as u32 + 4, error_discriminant(&e));
                        return;
                    }
                }
            };
            
            let retptr = retptr as u32;
            
            // Allocate memory for the result using realloc
            // For now, write ptr+len at retptr (caller should have allocated space)
            // This is a simplification - real implementation needs cabi_realloc
            write_u32(&mut caller, retptr, 0); // Ok
            // Write data to a fixed location (this is a hack - real impl needs malloc)
            let data_ptr = retptr + 8;
            write_memory(&mut caller, data_ptr, &data);
            write_u32(&mut caller, retptr + 4, data_ptr);
            write_u32(&mut caller, retptr + 8, data.len() as u32);
        },
    )?;
    
    // buffer-copy: func(src, src-offset, dst, dst-offset, size) -> result<_, gpu-error>
    // ABI: (src: i32, src_offset: i64, dst: i32, dst_offset: i64, size: i64, retptr: i32) -> ()
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-copy",
        move |mut caller: Caller<'_, T>, src: i32, src_offset: i64, dst: i32, dst_offset: i64, size: i64, retptr: i32| {
            debug!("[Host] buffer-copy(src={}, dst={}, size={})", src, dst, size);
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().buffer_copy(src as u32, src_offset as u64, dst as u32, dst_offset as u64, size as u64) {
                Ok(()) => {
                    write_u32(&mut caller, retptr, 0);
                }
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-destroy: func(buffer: buffer-id) -> result<_, gpu-error>
    // ABI: (buffer: i32, retptr: i32) -> ()
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-destroy",
        move |mut caller: Caller<'_, T>, buffer: i32, retptr: i32| {
            debug!("[Host] buffer-destroy(buffer={})", buffer);
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().buffer_destroy(buffer as u32) {
                Ok(()) => {
                    write_u32(&mut caller, retptr, 0);
                }
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // sync: func() -> ()
    // ABI: () -> ()
    linker.func_wrap(
        COMPUTE_MODULE,
        "sync",
        move |mut caller: Caller<'_, T>| {
            debug!("[Host] sync()");
            let state = get_state(caller.data_mut());
            state.backend().sync();
        },
    )?;
    
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/ml-kernels@0.1.0 interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // kernel-bootstrap-sample: func(params: bootstrap-params, output: buffer-id) -> result<_, gpu-error>
    // ABI: (params: i32, output: i32, retptr: i32) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-bootstrap-sample",
        move |mut caller: Caller<'_, T>, params_ptr: i32, output: i32, retptr: i32| {
            debug!("[Host] kernel-bootstrap-sample");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 12) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = BootstrapParams {
                n_samples: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                seed: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                max_index: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_bootstrap_sample(&params, output as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-find-split
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-find-split",
        move |mut caller: Caller<'_, T>, 
              params_ptr: i32,
              data: i32, labels: i32, indices: i32, thresholds: i32, output_scores: i32,
              retptr: i32| {
            debug!("[Host] kernel-find-split");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 16) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = FindSplitParams {
                n_samples: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_features: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                feature_idx: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                n_thresholds: u32::from_le_bytes([params_bytes[12], params_bytes[13], params_bytes[14], params_bytes[15]]),
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_find_split(&params, data as u32, labels as u32, indices as u32, thresholds as u32, output_scores as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-average
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-average",
        move |mut caller: Caller<'_, T>, params_ptr: i32, tree_predictions: i32, output: i32, retptr: i32| {
            debug!("[Host] kernel-average");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 8) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = AverageParams {
                n_trees: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_samples: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_average(&params, tree_predictions as u32, output as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-matmul
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-matmul",
        move |mut caller: Caller<'_, T>, params_ptr: i32, a: i32, b: i32, c: i32, retptr: i32| {
            debug!("[Host] kernel-matmul");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 28) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = MatmulParams {
                m: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                k: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                n: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                trans_a: params_bytes[12] != 0,
                trans_b: params_bytes[16] != 0,
                alpha: f32::from_le_bytes([params_bytes[20], params_bytes[21], params_bytes[22], params_bytes[23]]),
                beta: f32::from_le_bytes([params_bytes[24], params_bytes[25], params_bytes[26], params_bytes[27]]),
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_matmul(&params, a as u32, b as u32, c as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-elementwise
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-elementwise",
        move |mut caller: Caller<'_, T>, params_ptr: i32, input_a: i32, input_b: i32, output: i32, retptr: i32| {
            debug!("[Host] kernel-elementwise");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 8) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
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
                _ => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = ElementwiseParams { n_elements, op };
            let input_b_opt = if input_b == 0 { None } else { Some(input_b as u32) };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_elementwise(&params, input_a as u32, input_b_opt, output as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-reduce
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-reduce",
        move |mut caller: Caller<'_, T>, params_ptr: i32, input: i32, output: i32, retptr: i32| {
            debug!("[Host] kernel-reduce");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 8) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let n_elements = u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]);
            let op_code = u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]);
            
            let op = match op_code {
                0 => ReduceOp::Sum,
                1 => ReduceOp::Max,
                2 => ReduceOp::Min,
                3 => ReduceOp::Mean,
                4 => ReduceOp::Variance,
                _ => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = ReduceParams { n_elements, op };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_reduce(&params, input as u32, output as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-batch-predict
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-batch-predict",
        move |mut caller: Caller<'_, T>, 
              params_ptr: i32,
              samples: i32, tree_nodes: i32, tree_offsets: i32, output: i32,
              retptr: i32| {
            debug!("[Host] kernel-batch-predict");
            
            let params_bytes = match read_memory(&mut caller, params_ptr as u32, 16) {
                Some(b) => b,
                None => {
                    write_u32(&mut caller, retptr as u32, 1);
                    write_u32(&mut caller, retptr as u32 + 4, 4);
                    return;
                }
            };
            
            let params = BatchPredictParams {
                batch_size: u32::from_le_bytes([params_bytes[0], params_bytes[1], params_bytes[2], params_bytes[3]]),
                n_features: u32::from_le_bytes([params_bytes[4], params_bytes[5], params_bytes[6], params_bytes[7]]),
                n_trees: u32::from_le_bytes([params_bytes[8], params_bytes[9], params_bytes[10], params_bytes[11]]),
                max_depth: u32::from_le_bytes([params_bytes[12], params_bytes[13], params_bytes[14], params_bytes[15]]),
            };
            
            let state = get_state(caller.data_mut());
            let retptr = retptr as u32;
            
            match state.backend_mut().kernel_batch_predict(&params, samples as u32, tree_nodes as u32, tree_offsets as u32, output as u32) {
                Ok(()) => { write_u32(&mut caller, retptr, 0); },
                Err(e) => {
                    write_u32(&mut caller, retptr, 1);
                    write_u32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    info!("[Host] wasi:gpu host functions registered");
    info!("[Host]   - {} (buffer management, sync)", COMPUTE_MODULE);
    info!("[Host]   - {} (ML kernels)", ML_KERNELS_MODULE);
    
    Ok(())
}
