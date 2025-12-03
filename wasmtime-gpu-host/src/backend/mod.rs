//! GPU Backend Trait and Implementations
//!
//! Provides the `GpuBackend` trait and concrete implementations:
//! - `CudaBackend`: Uses CUDA/cuBLAS for NVIDIA GPUs (optimal for H100)
//! - `WebGpuBackend`: Uses wgpu/Vulkan for cross-platform support

pub mod cuda;
pub mod webgpu;

pub use cuda::CudaBackend;
pub use webgpu::WebGpuBackend;

use anyhow::Result;

/// Buffer usage flags (matching WIT definition)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage;

impl BufferUsage {
    pub const MAP_READ: u32 = 1 << 0;
    pub const MAP_WRITE: u32 = 1 << 1;
    pub const COPY_SRC: u32 = 1 << 2;
    pub const COPY_DST: u32 = 1 << 3;
    pub const UNIFORM: u32 = 1 << 4;
    pub const STORAGE: u32 = 1 << 5;
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub total_memory: u64,
    pub is_hardware: bool,
    pub compute_capability: String,
}

/// Error types for GPU operations
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Invalid buffer ID: {0}")]
    InvalidBuffer(u32),
    #[error("Kernel error: {0}")]
    KernelError(String),
    #[error("Device unavailable")]
    DeviceUnavailable,
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Backend error: {0}")]
    BackendError(String),
}

/// Buffer handle type
pub type BufferId = u32;

/// Bootstrap sampling parameters
#[derive(Debug, Clone, Copy)]
pub struct BootstrapParams {
    pub n_samples: u32,
    pub seed: u32,
    pub max_index: u32,
}

/// Find split parameters
#[derive(Debug, Clone, Copy)]
pub struct FindSplitParams {
    pub n_samples: u32,
    pub n_features: u32,
    pub feature_idx: u32,
    pub n_thresholds: u32,
}

/// Average parameters
#[derive(Debug, Clone, Copy)]
pub struct AverageParams {
    pub n_trees: u32,
    pub n_samples: u32,
}

/// Matrix multiplication parameters
#[derive(Debug, Clone, Copy)]
pub struct MatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
}

/// Elementwise operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    Relu,
    Sigmoid,
    Tanh,
    Add,
    Mul,
    Sqrt,
    Exp,
    Log,
}

/// Elementwise parameters
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseParams {
    pub n_elements: u32,
    pub op: ElementwiseOp,
}

/// Reduce operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
    Variance,
}

/// Reduce parameters
#[derive(Debug, Clone, Copy)]
pub struct ReduceParams {
    pub n_elements: u32,
    pub op: ReduceOp,
}

/// Batch prediction parameters
#[derive(Debug, Clone, Copy)]
pub struct BatchPredictParams {
    pub batch_size: u32,
    pub n_features: u32,
    pub n_trees: u32,
    pub max_depth: u32,
}

/// GPU Backend trait - implemented by both WebGPU and CUDA backends
pub trait GpuBackend: Send + Sync {
    /// Get device information
    fn device_info(&self) -> &DeviceInfo;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Buffer Management (compute interface)
    // ═══════════════════════════════════════════════════════════════════════
    
    /// Create a new GPU buffer
    fn buffer_create(&mut self, size: u64, usage: u32) -> Result<BufferId, GpuError>;
    
    /// Write data to a GPU buffer
    fn buffer_write(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), GpuError>;
    
    /// Read data from a GPU buffer
    fn buffer_read(&self, buffer: BufferId, offset: u64, size: u32) -> Result<Vec<u8>, GpuError>;
    
    /// Copy data between GPU buffers
    fn buffer_copy(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: BufferId,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuError>;
    
    /// Destroy a GPU buffer
    fn buffer_destroy(&mut self, buffer: BufferId) -> Result<(), GpuError>;
    
    /// Synchronize GPU operations
    fn sync(&self);
    
    // ═══════════════════════════════════════════════════════════════════════
    // ML Kernels (ml-kernels interface)
    // ═══════════════════════════════════════════════════════════════════════
    
    /// Bootstrap sampling
    fn kernel_bootstrap_sample(
        &mut self,
        params: &BootstrapParams,
        output: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Find best split for decision tree
    fn kernel_find_split(
        &mut self,
        params: &FindSplitParams,
        data: BufferId,
        labels: BufferId,
        indices: BufferId,
        thresholds: BufferId,
        output_scores: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Average tree predictions
    fn kernel_average(
        &mut self,
        params: &AverageParams,
        tree_predictions: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Matrix multiplication
    fn kernel_matmul(
        &mut self,
        params: &MatmulParams,
        a: BufferId,
        b: BufferId,
        c: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Elementwise operation
    fn kernel_elementwise(
        &mut self,
        params: &ElementwiseParams,
        input_a: BufferId,
        input_b: Option<BufferId>,
        output: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Reduce operation
    fn kernel_reduce(
        &mut self,
        params: &ReduceParams,
        input: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError>;
    
    /// Batch prediction for Random Forest
    fn kernel_batch_predict(
        &mut self,
        params: &BatchPredictParams,
        samples: BufferId,
        tree_nodes: BufferId,
        tree_offsets: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError>;
}
