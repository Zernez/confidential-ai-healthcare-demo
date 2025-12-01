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
// ═══════════════════════════════════════════════════════════════════════════

/// Get information about the GPU device
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("get-device-info")))
void wasi_gpu_get_device_info(
    char* name_ptr, uint32_t name_len,
    char* backend_ptr, uint32_t backend_len,
    uint64_t* total_memory,
    uint32_t* is_hardware,
    char* compute_cap_ptr, uint32_t compute_cap_len
);

/// Create a new GPU buffer
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-create")))
wasi_gpu_error wasi_gpu_buffer_create(
    uint64_t size,
    uint32_t usage,
    wasi_gpu_buffer_id* out_buffer_id
);

/// Write data to a GPU buffer
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-write")))
wasi_gpu_error wasi_gpu_buffer_write(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    const uint8_t* data,
    uint32_t data_len
);

/// Read data from a GPU buffer
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-read")))
wasi_gpu_error wasi_gpu_buffer_read(
    wasi_gpu_buffer_id buffer,
    uint64_t offset,
    uint32_t size,
    uint8_t* out_data,
    uint32_t* out_data_len
);

/// Copy data between GPU buffers
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-copy")))
wasi_gpu_error wasi_gpu_buffer_copy(
    wasi_gpu_buffer_id src,
    uint64_t src_offset,
    wasi_gpu_buffer_id dst,
    uint64_t dst_offset,
    uint64_t size
);

/// Destroy a GPU buffer
__attribute__((import_module("wasi:gpu/compute@0.1.0")))
__attribute__((import_name("buffer-destroy")))
wasi_gpu_error wasi_gpu_buffer_destroy(wasi_gpu_buffer_id buffer);

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

/// Generate bootstrap sample indices on GPU
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-bootstrap-sample")))
wasi_gpu_error wasi_gpu_kernel_bootstrap_sample(
    const wasi_gpu_bootstrap_params_t* params,
    wasi_gpu_buffer_id output_indices
);

/// Find the best split threshold for a decision tree node
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-find-split")))
wasi_gpu_error wasi_gpu_kernel_find_split(
    const wasi_gpu_find_split_params_t* params,
    wasi_gpu_buffer_id data,
    wasi_gpu_buffer_id labels,
    wasi_gpu_buffer_id indices,
    wasi_gpu_buffer_id thresholds,
    wasi_gpu_buffer_id output_scores
);

/// Average predictions across all trees in a forest
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-average")))
wasi_gpu_error wasi_gpu_kernel_average(
    const wasi_gpu_average_params_t* params,
    wasi_gpu_buffer_id tree_predictions,
    wasi_gpu_buffer_id output
);

/// Matrix multiplication
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-matmul")))
wasi_gpu_error wasi_gpu_kernel_matmul(
    const wasi_gpu_matmul_params_t* params,
    wasi_gpu_buffer_id a,
    wasi_gpu_buffer_id b,
    wasi_gpu_buffer_id c
);

/// Apply element-wise operation
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-elementwise")))
wasi_gpu_error wasi_gpu_kernel_elementwise(
    const wasi_gpu_elementwise_params_t* params,
    wasi_gpu_buffer_id input_a,
    wasi_gpu_buffer_id input_b,  // 0 if not used (unary op)
    wasi_gpu_buffer_id output
);

/// Reduce array to single value
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-reduce")))
wasi_gpu_error wasi_gpu_kernel_reduce(
    const wasi_gpu_reduce_params_t* params,
    wasi_gpu_buffer_id input,
    wasi_gpu_buffer_id output
);

/// Batch prediction for Random Forest
__attribute__((import_module("wasi:gpu/ml-kernels@0.1.0")))
__attribute__((import_name("kernel-batch-predict")))
wasi_gpu_error wasi_gpu_kernel_batch_predict(
    const wasi_gpu_batch_predict_params_t* params,
    wasi_gpu_buffer_id samples,
    wasi_gpu_buffer_id tree_nodes,
    wasi_gpu_buffer_id tree_offsets,
    wasi_gpu_buffer_id output
);

#ifdef __cplusplus
}
#endif

#endif // WASI_GPU_H
