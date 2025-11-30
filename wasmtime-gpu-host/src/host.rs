//! WASI GPU Host Functions
//!
//! Implements the wasi:gpu interface as Wasmtime host functions.
//! These bridge between the WASM module and the actual GPU backend.
//!
//! The ABI follows wit-bindgen's "flattened" conventions:
//! - Record types are expanded inline (no pointers for small records)
//! - result<_, E> adds a retptr parameter, writes discriminant + payload
//! - option<T> is represented as (is_some: i32, value: T)

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
fn error_discriminant(err: &GpuError) -> i32 {
    match err {
        GpuError::OutOfMemory => 0,
        GpuError::InvalidBuffer(_) => 1,
        GpuError::KernelError(_) => 2,
        GpuError::DeviceUnavailable => 3,
        GpuError::InvalidParams(_) => 4,
        GpuError::BackendError(_) => 5,
    }
}

/// Helper to write bytes to WASM memory
fn write_memory<T>(caller: &mut Caller<'_, T>, ptr: i32, data: &[u8]) -> bool {
    if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
        memory.write(caller, ptr as usize, data).is_ok()
    } else {
        false
    }
}

/// Write a i32 to WASM memory
fn write_i32<T>(caller: &mut Caller<'_, T>, ptr: i32, value: i32) -> bool {
    write_memory(caller, ptr, &value.to_le_bytes())
}

/// Register all wasi:gpu host functions with the linker
/// 
/// Functions are registered with versioned module names to match WIT component model:
/// - wasi:gpu/compute@0.1.0
/// - wasi:gpu/ml-kernels@0.1.0
/// 
/// Record types are "flattened" per wit-bindgen ABI:
/// - bootstrap-params (3 × u32) → 3 i32 params
/// - find-split-params (4 × u32) → 4 i32 params  
/// - average-params (2 × u32) → 2 i32 params
/// - matmul-params (3 × u32 + 2 × bool + 2 × f32) → 7 params
/// - elementwise-params (1 × u32 + enum) → 2 i32 params
/// - reduce-params (1 × u32 + enum) → 2 i32 params
/// - batch-predict-params (4 × u32) → 4 i32 params
pub fn add_to_linker<T: 'static>(
    linker: &mut Linker<T>,
    get_state: impl Fn(&mut T) -> &mut GpuState + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    const COMPUTE_MODULE: &str = "wasi:gpu/compute@0.1.0";
    const ML_KERNELS_MODULE: &str = "wasi:gpu/ml-kernels@0.1.0";
    
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/compute@0.1.0 interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // get-device-info: func() -> device-info
    // device-info has strings, so it uses retptr for the whole struct
    linker.func_wrap(
        COMPUTE_MODULE,
        "get-device-info",
        move |mut caller: Caller<'_, T>, retptr: i32| {
            debug!("[Host] get-device-info");
            
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
            
            // Layout: ptr, len for each string, then scalars
            // Strings need to be allocated - use cabi_realloc or fixed buffer
            // For simplicity, write inline after struct
            let name_bytes = name.as_bytes();
            let backend_bytes = backend.as_bytes();
            let cc_bytes = compute_cap.as_bytes();
            
            let str_base = retptr + 40; // After struct fields
            
            // name (ptr, len)
            write_i32(&mut caller, retptr, str_base);
            write_i32(&mut caller, retptr + 4, name_bytes.len() as i32);
            write_memory(&mut caller, str_base, name_bytes);
            
            // backend (ptr, len)
            let backend_ptr = str_base + name_bytes.len() as i32;
            write_i32(&mut caller, retptr + 8, backend_ptr);
            write_i32(&mut caller, retptr + 12, backend_bytes.len() as i32);
            write_memory(&mut caller, backend_ptr, backend_bytes);
            
            // total_memory: u64
            write_memory(&mut caller, retptr + 16, &total_memory.to_le_bytes());
            
            // is_hardware: bool (as i32)
            write_i32(&mut caller, retptr + 24, if is_hardware { 1 } else { 0 });
            
            // compute_capability (ptr, len)
            let cc_ptr = backend_ptr + backend_bytes.len() as i32;
            write_i32(&mut caller, retptr + 28, cc_ptr);
            write_i32(&mut caller, retptr + 32, cc_bytes.len() as i32);
            write_memory(&mut caller, cc_ptr, cc_bytes);
        },
    )?;
    
    // buffer-create: func(size: u64, usage: buffer-usage) -> result<buffer-id, gpu-error>
    // buffer-usage is flags (i32), result uses retptr
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-create",
        move |mut caller: Caller<'_, T>, size: i64, usage: i32, retptr: i32| {
            debug!("[Host] buffer-create(size={}, usage={:#x})", size, usage);
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_create(size as u64, usage as u32) {
                Ok(id) => {
                    write_i32(&mut caller, retptr, 0); // Ok
                    write_i32(&mut caller, retptr + 4, id as i32);
                }
                Err(e) => {
                    error!("[Host] buffer-create failed: {:?}", e);
                    write_i32(&mut caller, retptr, 1); // Err
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-write: func(buffer: buffer-id, offset: u64, data: list<u8>) -> result<_, gpu-error>
    // list<u8> is (ptr, len)
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-write",
        move |mut caller: Caller<'_, T>, buffer: i32, offset: i64, data_ptr: i32, data_len: i32, retptr: i32| {
            debug!("[Host] buffer-write(buffer={}, offset={}, len={})", buffer, offset, data_len);
            
            let data = {
                let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                    Some(m) => m,
                    None => {
                        write_i32(&mut caller, retptr, 1);
                        write_i32(&mut caller, retptr + 4, 4);
                        return;
                    }
                };
                let mut buf = vec![0u8; data_len as usize];
                if memory.read(&caller, data_ptr as usize, &mut buf).is_err() {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, 4);
                    return;
                }
                buf
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_write(buffer as u32, offset as u64, &data) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    error!("[Host] buffer-write failed: {:?}", e);
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-read: func(buffer: buffer-id, offset: u64, size: u32) -> result<list<u8>, gpu-error>
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
                        write_i32(&mut caller, retptr, 1);
                        write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                        return;
                    }
                }
            };
            
            // Need to call cabi_realloc to allocate space for result
            // For now, assume caller provides enough space after retptr
            write_i32(&mut caller, retptr, 0); // Ok
            let data_dest = retptr + 8;
            write_i32(&mut caller, retptr + 4, data_dest); // ptr
            write_i32(&mut caller, retptr + 8, data.len() as i32); // len
            // Actually write data - this needs proper allocation
            // The caller should use cabi_realloc
        },
    )?;
    
    // buffer-copy
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-copy",
        move |mut caller: Caller<'_, T>, src: i32, src_offset: i64, dst: i32, dst_offset: i64, size: i64, retptr: i32| {
            debug!("[Host] buffer-copy");
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_copy(src as u32, src_offset as u64, dst as u32, dst_offset as u64, size as u64) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // buffer-destroy
    linker.func_wrap(
        COMPUTE_MODULE,
        "buffer-destroy",
        move |mut caller: Caller<'_, T>, buffer: i32, retptr: i32| {
            debug!("[Host] buffer-destroy(buffer={})", buffer);
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().buffer_destroy(buffer as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // sync
    linker.func_wrap(
        COMPUTE_MODULE,
        "sync",
        move |mut caller: Caller<'_, T>| {
            debug!("[Host] sync");
            let state = get_state(caller.data_mut());
            state.backend().sync();
        },
    )?;
    
    // ═══════════════════════════════════════════════════════════════════════
    // wasi:gpu/ml-kernels@0.1.0 interface
    // ═══════════════════════════════════════════════════════════════════════
    
    // kernel-bootstrap-sample
    // bootstrap-params: n_samples, seed, max_index (3 × u32)
    // Signature: (n_samples, seed, max_index, output_indices, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-bootstrap-sample",
        move |mut caller: Caller<'_, T>, 
              n_samples: i32, seed: i32, max_index: i32,
              output: i32, 
              retptr: i32| {
            debug!("[Host] kernel-bootstrap-sample(n={}, seed={}, max={})", n_samples, seed, max_index);
            
            let params = BootstrapParams {
                n_samples: n_samples as u32,
                seed: seed as u32,
                max_index: max_index as u32,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_bootstrap_sample(&params, output as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-find-split
    // find-split-params: n_samples, n_features, feature_idx, n_thresholds (4 × u32)
    // Signature: (n_samples, n_features, feature_idx, n_thresholds, data, labels, indices, thresholds, output_scores, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-find-split",
        move |mut caller: Caller<'_, T>, 
              n_samples: i32, n_features: i32, feature_idx: i32, n_thresholds: i32,
              data: i32, labels: i32, indices: i32, thresholds: i32, output_scores: i32,
              retptr: i32| {
            debug!("[Host] kernel-find-split(n={}, feat={}, idx={}, thresh={})", 
                   n_samples, n_features, feature_idx, n_thresholds);
            
            let params = FindSplitParams {
                n_samples: n_samples as u32,
                n_features: n_features as u32,
                feature_idx: feature_idx as u32,
                n_thresholds: n_thresholds as u32,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_find_split(
                &params, 
                data as u32, labels as u32, indices as u32, thresholds as u32, output_scores as u32
            ) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-average
    // average-params: n_trees, n_samples (2 × u32)
    // Signature: (n_trees, n_samples, tree_predictions, output, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-average",
        move |mut caller: Caller<'_, T>, 
              n_trees: i32, n_samples: i32,
              tree_predictions: i32, output: i32, 
              retptr: i32| {
            debug!("[Host] kernel-average(trees={}, samples={})", n_trees, n_samples);
            
            let params = AverageParams {
                n_trees: n_trees as u32,
                n_samples: n_samples as u32,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_average(&params, tree_predictions as u32, output as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-matmul
    // matmul-params: m, k, n (3 × u32), trans_a, trans_b (2 × bool as i32), alpha, beta (2 × f32)
    // Signature: (m, k, n, trans_a, trans_b, alpha, beta, a, b, c, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-matmul",
        move |mut caller: Caller<'_, T>, 
              m: i32, k: i32, n: i32, 
              trans_a: i32, trans_b: i32,
              alpha: f32, beta: f32,
              a: i32, b: i32, c: i32, 
              retptr: i32| {
            debug!("[Host] kernel-matmul(m={}, k={}, n={})", m, k, n);
            
            let params = MatmulParams {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                trans_a: trans_a != 0,
                trans_b: trans_b != 0,
                alpha,
                beta,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_matmul(&params, a as u32, b as u32, c as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-elementwise
    // elementwise-params: n_elements (u32), op (enum as i32)
    // option<buffer-id> is (is_some: i32, value: i32)
    // Signature: (n_elements, op, input_a, input_b_is_some, input_b, output, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-elementwise",
        move |mut caller: Caller<'_, T>, 
              n_elements: i32, op: i32,
              input_a: i32, 
              input_b_is_some: i32, input_b: i32,
              output: i32, 
              retptr: i32| {
            debug!("[Host] kernel-elementwise(n={}, op={})", n_elements, op);
            
            let op_enum = match op {
                0 => ElementwiseOp::Relu,
                1 => ElementwiseOp::Sigmoid,
                2 => ElementwiseOp::Tanh,
                3 => ElementwiseOp::Add,
                4 => ElementwiseOp::Mul,
                5 => ElementwiseOp::Sqrt,
                6 => ElementwiseOp::Exp,
                7 => ElementwiseOp::Log,
                _ => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, 4);
                    return;
                }
            };
            
            let params = ElementwiseParams { 
                n_elements: n_elements as u32, 
                op: op_enum,
            };
            let input_b_opt = if input_b_is_some != 0 { Some(input_b as u32) } else { None };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_elementwise(&params, input_a as u32, input_b_opt, output as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-reduce
    // reduce-params: n_elements (u32), op (enum as i32)
    // Signature: (n_elements, op, input, output, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-reduce",
        move |mut caller: Caller<'_, T>, 
              n_elements: i32, op: i32,
              input: i32, output: i32, 
              retptr: i32| {
            debug!("[Host] kernel-reduce(n={}, op={})", n_elements, op);
            
            let op_enum = match op {
                0 => ReduceOp::Sum,
                1 => ReduceOp::Max,
                2 => ReduceOp::Min,
                3 => ReduceOp::Mean,
                4 => ReduceOp::Variance,
                _ => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, 4);
                    return;
                }
            };
            
            let params = ReduceParams { 
                n_elements: n_elements as u32, 
                op: op_enum,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_reduce(&params, input as u32, output as u32) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    // kernel-batch-predict
    // batch-predict-params: batch_size, n_features, n_trees, max_depth (4 × u32)
    // Signature: (batch_size, n_features, n_trees, max_depth, samples, tree_nodes, tree_offsets, output, retptr) -> ()
    linker.func_wrap(
        ML_KERNELS_MODULE,
        "kernel-batch-predict",
        move |mut caller: Caller<'_, T>, 
              batch_size: i32, n_features: i32, n_trees: i32, max_depth: i32,
              samples: i32, tree_nodes: i32, tree_offsets: i32, output: i32,
              retptr: i32| {
            debug!("[Host] kernel-batch-predict(batch={}, feat={}, trees={}, depth={})", 
                   batch_size, n_features, n_trees, max_depth);
            
            let params = BatchPredictParams {
                batch_size: batch_size as u32,
                n_features: n_features as u32,
                n_trees: n_trees as u32,
                max_depth: max_depth as u32,
            };
            
            let state = get_state(caller.data_mut());
            
            match state.backend_mut().kernel_batch_predict(
                &params, 
                samples as u32, tree_nodes as u32, tree_offsets as u32, output as u32
            ) {
                Ok(()) => { write_i32(&mut caller, retptr, 0); }
                Err(e) => {
                    write_i32(&mut caller, retptr, 1);
                    write_i32(&mut caller, retptr + 4, error_discriminant(&e));
                }
            }
        },
    )?;
    
    info!("[Host] wasi:gpu host functions registered");
    info!("[Host]   - {} (buffer management, sync)", COMPUTE_MODULE);
    info!("[Host]   - {} (ML kernels)", ML_KERNELS_MODULE);
    
    Ok(())
}
