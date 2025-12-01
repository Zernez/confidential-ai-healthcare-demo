/// WASI GPU Interface - C++ binding header
/// 
/// This file provides C++ declarations for wasi:gpu host functions.
/// These are implemented by the wasmtime-gpu-host runtime.

#ifndef WASI_GPU_H
#define WASI_GPU_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Buffer Usage Flags (matching WIT flags buffer-usage)
// ═══════════════════════════════════════════════════════════════════════════

#define WASI_GPU_BUFFER_USAGE_MAP_READ   (1 << 0)
#define WASI_GPU_BUFFER_USAGE_MAP_WRITE  (1 << 1)
#define WASI_GPU_BUFFER_USAGE_COPY_SRC   (1 << 2)
#define WASI_GPU_BUFFER_USAGE_COPY_DST   (1 << 3)
#define WASI_GPU_BUFFER_USAGE_UNIFORM    (1 << 4)
#define WASI_GPU_BUFFER_USAGE_STORAGE    (1 << 5)

// ═══════════════════════════════════════════════════════════════════════════
// Error Codes
// ═══════════════════════════════════════════════════════════════════════════

#define WASI_GPU_SUCCESS              0
#define WASI_GPU_ERROR_OUT_OF_MEMORY  1
#define WASI_GPU_ERROR_INVALID_BUFFER 2
#define WASI_GPU_ERROR_KERNEL_ERROR   3
#define WASI_GPU_ERROR_DEVICE_UNAVAIL 4
#define WASI_GPU_ERROR_INVALID_PARAMS 5

// ═══════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════

typedef uint32_t wasi_gpu_buffer_id;
typedef uint32_t wasi_gpu_error;

// ═══════════════════════════════════════════════════════════════════════════
// Device Info Structure
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    char name[256];
    char backend[64];
    uint64_t total_memory;
    uint32_t is_hardware;
    char compute_capability[32];
} wasi_gpu_device_info_t;

// ═══════════════════════════════════════════════════════════════════════════
// compute Interface - Low-level GPU Operations
// 
// IMPORTANT: These functions use wit-bindgen ABI conventions:
// - result<T, E> uses a retptr parameter for returning values
// - Strings in results are allocated via cabi_realloc and returned as (ptr, len)
// - Records are "flattened" as consecutive parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Device info result structure (parsed from retptr)
/// Layout at retptr: name_ptr(4), name_len(4), backend_ptr(4), backend_len(4),
///                   total_memory(8), is_hardware(4), cc_ptr(4), cc_len(4) = 36 bytes
typedef struct {
    uint32_t name_ptr;
    uint32_t name_len;
    uint32_t backend_ptr;
    uint32_t backend_len;
    uint64_t total_memory;
    uint32_t is_hardware;
    uint32_t compute_cap_ptr;
    uint32_t compute_cap_len;
} wasi_gpu_device_info_raw_t;

/// Get information about the GPU device (raw ABI - returns data at retptr)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("get-device-info")))
void __wasi_gpu_get_device_info_raw(wasi_gpu_device_info_raw_t* retptr);

/// Wrapper to get device info in a convenient C structure
static inline void wasi_gpu_get_device_info(wasi_gpu_device_info_t* info) {
    wasi_gpu_device_info_raw_t raw;
    __wasi_gpu_get_device_info_raw(&raw);
    
    // Copy strings from WASM linear memory pointers
    if (raw.name_ptr != 0 && raw.name_len > 0 && raw.name_len < sizeof(info->name)) {
        const char* src = (const char*)(uintptr_t)raw.name_ptr;
        for (uint32_t i = 0; i < raw.name_len; ++i) {
            info->name[i] = src[i];
        }
        info->name[raw.name_len] = '\0';
    } else {
        info->name[0] = '\0';
    }
    
    if (raw.backend_ptr != 0 && raw.backend_len > 0 && raw.backend_len < sizeof(info->backend)) {
        const char* src = (const char*)(uintptr_t)raw.backend_ptr;
        for (uint32_t i = 0; i < raw.backend_len; ++i) {
            info->backend[i] = src[i];
        }
        info->backend[raw.backend_len] = '\0';
    } else {
        info->backend[0] = '\0';
    }
    
    if (raw.compute_cap_ptr != 0 && raw.compute_cap_len > 0 && raw.compute_cap_len < sizeof(info->compute_capability)) {
        const char* src = (const char*)(uintptr_t)raw.compute_cap_ptr;
        for (uint32_t i = 0; i < raw.compute_cap_len; ++i) {
            info->compute_capability[i] = src[i];
        }
        info->compute_capability[raw.compute_cap_len] = '\0';
    } else {
        info->compute_capability[0] = '\0';
    }
    
    info->total_memory = raw.total_memory;
    info->is_hardware = raw.is_hardware;
}

/// Result structure for buffer operations
/// Layout: discriminant(4) + payload(variable)
typedef struct {
    uint32_t is_err;      // 0 = Ok, 1 = Err
    uint32_t value;       // buffer_id on success, error code on failure
} wasi_gpu_result_buffer_t;

/// Result structure for list<u8> (e.g., buffer-read)
/// Layout: discriminant(4) + ptr(4) + len(4) on success, or discriminant(4) + error(4) on failure
typedef struct {
    uint32_t is_err;
    uint32_t data_ptr;
    uint32_t data_len;
} wasi_gpu_result_list_t;

/// Create a new GPU buffer (raw ABI)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-create")))
void __wasi_gpu_buffer_create_raw(uint64_t size, uint32_t usage, wasi_gpu_result_buffer_t* retptr);

/// Wrapper for buffer creation
static inline wasi_gpu_error wasi_gpu_buffer_create(
    uint64_t size,
    uint32_t usage,
    wasi_gpu_buffer_id* out_buffer_id
) {
    wasi_gpu_result_buffer_t result;
    __wasi_gpu_buffer_create_raw(size, usage, &result);
    if (result.is_err) {
        return (wasi_gpu_error)(result.value + 1); // Convert to our error codes
    }
    *out_buffer_id = result.value;
    return WASI_GPU_SUCCESS;
}

/// Result for void operations (only error possible)
typedef struct {
    uint32_t is_err;
    uint32_t error_code;
} wasi_gpu_result_void_t;

/// Write data to a GPU buffer (raw ABI)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-write")))
void __wasi_gpu_buffer_write_raw(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    const uint8_t* data,
    uint32_t data_len,
    wasi_gpu_result_void_t* retptr
);

/// Wrapper for buffer write
static inline wasi_gpu_error wasi_gpu_buffer_write(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    const uint8_t* data,
    uint32_t data_len
) {
    wasi_gpu_result_void_t result;
    __wasi_gpu_buffer_write_raw(buffer, offset, data, data_len, &result);
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Read data from a GPU buffer (raw ABI)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-read")))
void __wasi_gpu_buffer_read_raw(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    uint32_t size,
    wasi_gpu_result_list_t* retptr
);

/// Wrapper for buffer read
static inline wasi_gpu_error wasi_gpu_buffer_read(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    uint32_t size,
    uint8_t* out_data,
    uint32_t* out_data_len
) {
    wasi_gpu_result_list_t result;
    __wasi_gpu_buffer_read_raw(buffer, offset, size, &result);
    if (result.is_err) {
        return (wasi_gpu_error)(result.data_ptr + 1); // error code is in data_ptr field for errors
    }
    // Copy data from returned pointer
    const uint8_t* src = (const uint8_t*)(uintptr_t)result.data_ptr;
    for (uint32_t i = 0; i < result.data_len && i < size; ++i) {
        out_data[i] = src[i];
    }
    *out_data_len = result.data_len;
    return WASI_GPU_SUCCESS;
}

/// Copy data between GPU buffers (raw ABI)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-copy")))
void __wasi_gpu_buffer_copy_raw(
    wasi_gpu_buffer_id src,
    uint64_t src_offset,
    wasi_gpu_buffer_id dst,
    uint64_t dst_offset,
    uint64_t size,
    wasi_gpu_result_void_t* retptr
);

/// Wrapper for buffer copy
static inline wasi_gpu_error wasi_gpu_buffer_copy(
    wasi_gpu_buffer_id src,
    uint64_t src_offset,
    wasi_gpu_buffer_id dst,
    uint64_t dst_offset,
    uint64_t size
) {
    wasi_gpu_result_void_t result;
    __wasi_gpu_buffer_copy_raw(src, src_offset, dst, dst_offset, size, &result);
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Destroy a GPU buffer (raw ABI)
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-destroy")))
void __wasi_gpu_buffer_destroy_raw(wasi_gpu_buffer_id buffer, wasi_gpu_result_void_t* retptr);

/// Wrapper for buffer destroy
static inline wasi_gpu_error wasi_gpu_buffer_destroy(wasi_gpu_buffer_id buffer) {
    wasi_gpu_result_void_t result;
    __wasi_gpu_buffer_destroy_raw(buffer, &result);
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Synchronize GPU operations
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("sync")))
void wasi_gpu_sync(void);

// ═══════════════════════════════════════════════════════════════════════════
// ml-kernels Interface - High-level ML Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Bootstrap sampling parameters
typedef struct {
    uint32_t n_samples;
    uint32_t seed;
    uint32_t max_index;
} wasi_gpu_bootstrap_params_t;

/// Find split parameters
typedef struct {
    uint32_t n_samples;
    uint32_t n_features;
    uint32_t feature_idx;
    uint32_t n_thresholds;
} wasi_gpu_find_split_params_t;

/// Average parameters
typedef struct {
    uint32_t n_trees;
    uint32_t n_samples;
} wasi_gpu_average_params_t;

/// Matrix multiplication parameters
typedef struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t trans_a;  // bool as uint32
    uint32_t trans_b;  // bool as uint32
    float alpha;
    float beta;
} wasi_gpu_matmul_params_t;

/// Elementwise operation type
typedef enum {
    WASI_GPU_ELEMENTWISE_RELU = 0,
    WASI_GPU_ELEMENTWISE_SIGMOID = 1,
    WASI_GPU_ELEMENTWISE_TANH = 2,
    WASI_GPU_ELEMENTWISE_ADD = 3,
    WASI_GPU_ELEMENTWISE_MUL = 4,
    WASI_GPU_ELEMENTWISE_SQRT = 5,
    WASI_GPU_ELEMENTWISE_EXP = 6,
    WASI_GPU_ELEMENTWISE_LOG = 7,
} wasi_gpu_elementwise_op_t;

/// Elementwise parameters
typedef struct {
    uint32_t n_elements;
    wasi_gpu_elementwise_op_t op;
} wasi_gpu_elementwise_params_t;

/// Reduce operation type
typedef enum {
    WASI_GPU_REDUCE_SUM = 0,
    WASI_GPU_REDUCE_MAX = 1,
    WASI_GPU_REDUCE_MIN = 2,
    WASI_GPU_REDUCE_MEAN = 3,
    WASI_GPU_REDUCE_VARIANCE = 4,
} wasi_gpu_reduce_op_t;

/// Reduce parameters
typedef struct {
    uint32_t n_elements;
    wasi_gpu_reduce_op_t op;
} wasi_gpu_reduce_params_t;

/// Batch prediction parameters
typedef struct {
    uint32_t batch_size;
    uint32_t n_features;
    uint32_t n_trees;
    uint32_t max_depth;
} wasi_gpu_batch_predict_params_t;

// ═══════════════════════════════════════════════════════════════════════════
// ML Kernel Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Result for ML kernel operations
typedef wasi_gpu_result_void_t wasi_gpu_result_kernel_t;

/// Generate bootstrap sample indices on GPU (raw ABI - params are flattened)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-bootstrap-sample")))
void __wasi_gpu_kernel_bootstrap_sample_raw(
    uint32_t n_samples,
    uint32_t seed,
    uint32_t max_index,
    wasi_gpu_buffer_id output_indices,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel bootstrap sample
static inline wasi_gpu_error wasi_gpu_kernel_bootstrap_sample(
    const wasi_gpu_bootstrap_params_t* params,
    wasi_gpu_buffer_id output_indices
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_bootstrap_sample_raw(
        params->n_samples, params->seed, params->max_index,
        output_indices, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Find the best split threshold for a decision tree node (raw ABI)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-find-split")))
void __wasi_gpu_kernel_find_split_raw(
    uint32_t n_samples,
    uint32_t n_features,
    uint32_t feature_idx,
    uint32_t n_thresholds,
    wasi_gpu_buffer_id data,
    wasi_gpu_buffer_id labels,
    wasi_gpu_buffer_id indices,
    wasi_gpu_buffer_id thresholds,
    wasi_gpu_buffer_id output_scores,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel find split
static inline wasi_gpu_error wasi_gpu_kernel_find_split(
    const wasi_gpu_find_split_params_t* params,
    wasi_gpu_buffer_id data,
    wasi_gpu_buffer_id labels,
    wasi_gpu_buffer_id indices,
    wasi_gpu_buffer_id thresholds,
    wasi_gpu_buffer_id output_scores
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_find_split_raw(
        params->n_samples, params->n_features, params->feature_idx, params->n_thresholds,
        data, labels, indices, thresholds, output_scores, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Average predictions across all trees in a forest (raw ABI)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-average")))
void __wasi_gpu_kernel_average_raw(
    uint32_t n_trees,
    uint32_t n_samples,
    wasi_gpu_buffer_id tree_predictions,
    wasi_gpu_buffer_id output,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel average
static inline wasi_gpu_error wasi_gpu_kernel_average(
    const wasi_gpu_average_params_t* params,
    wasi_gpu_buffer_id tree_predictions,
    wasi_gpu_buffer_id output
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_average_raw(
        params->n_trees, params->n_samples,
        tree_predictions, output, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Matrix multiplication (raw ABI)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-matmul")))
void __wasi_gpu_kernel_matmul_raw(
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t trans_a,
    uint32_t trans_b,
    float alpha,
    float beta,
    wasi_gpu_buffer_id a,
    wasi_gpu_buffer_id b,
    wasi_gpu_buffer_id c,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel matmul
static inline wasi_gpu_error wasi_gpu_kernel_matmul(
    const wasi_gpu_matmul_params_t* params,
    wasi_gpu_buffer_id a,
    wasi_gpu_buffer_id b,
    wasi_gpu_buffer_id c
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_matmul_raw(
        params->m, params->k, params->n,
        params->trans_a, params->trans_b,
        params->alpha, params->beta,
        a, b, c, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Apply element-wise operation (raw ABI with option<buffer-id> as is_some + value)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-elementwise")))
void __wasi_gpu_kernel_elementwise_raw(
    uint32_t n_elements,
    uint32_t op,
    wasi_gpu_buffer_id input_a,
    uint32_t input_b_is_some,
    wasi_gpu_buffer_id input_b,
    wasi_gpu_buffer_id output,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel elementwise
static inline wasi_gpu_error wasi_gpu_kernel_elementwise(
    const wasi_gpu_elementwise_params_t* params,
    wasi_gpu_buffer_id input_a,
    wasi_gpu_buffer_id input_b,  // 0 if not used
    wasi_gpu_buffer_id output
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_elementwise_raw(
        params->n_elements, (uint32_t)params->op,
        input_a,
        (input_b != 0) ? 1 : 0, input_b,
        output, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Reduce array to single value (raw ABI)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-reduce")))
void __wasi_gpu_kernel_reduce_raw(
    uint32_t n_elements,
    uint32_t op,
    wasi_gpu_buffer_id input,
    wasi_gpu_buffer_id output,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel reduce
static inline wasi_gpu_error wasi_gpu_kernel_reduce(
    const wasi_gpu_reduce_params_t* params,
    wasi_gpu_buffer_id input,
    wasi_gpu_buffer_id output
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_reduce_raw(
        params->n_elements, (uint32_t)params->op,
        input, output, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

/// Batch prediction for Random Forest (raw ABI)
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-batch-predict")))
void __wasi_gpu_kernel_batch_predict_raw(
    uint32_t batch_size,
    uint32_t n_features,
    uint32_t n_trees,
    uint32_t max_depth,
    wasi_gpu_buffer_id samples,
    wasi_gpu_buffer_id tree_nodes,
    wasi_gpu_buffer_id tree_offsets,
    wasi_gpu_buffer_id output,
    wasi_gpu_result_kernel_t* retptr
);

/// Wrapper for kernel batch predict
static inline wasi_gpu_error wasi_gpu_kernel_batch_predict(
    const wasi_gpu_batch_predict_params_t* params,
    wasi_gpu_buffer_id samples,
    wasi_gpu_buffer_id tree_nodes,
    wasi_gpu_buffer_id tree_offsets,
    wasi_gpu_buffer_id output
) {
    wasi_gpu_result_kernel_t result;
    __wasi_gpu_kernel_batch_predict_raw(
        params->batch_size, params->n_features, params->n_trees, params->max_depth,
        samples, tree_nodes, tree_offsets, output, &result
    );
    if (result.is_err) {
        return (wasi_gpu_error)(result.error_code + 1);
    }
    return WASI_GPU_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif // WASI_GPU_H
