/**
 * GPU Backend using wgpu
 * 
 * Provides actual GPU access through wgpu library.
 * Optimized for headless server environments (like Azure VMs with H100).
 * 
 * NOTE: NVIDIA datacenter drivers (like 580.x) may not include full Vulkan support.
 * In such cases, compute workloads should use CUDA directly instead of WebGPU.
 */

use anyhow::Result;
use wgpu::*;
use std::sync::Arc;
use log::{info, warn, debug, error};

/// GPU availability status
#[derive(Debug, Clone)]
pub enum GpuStatus {
    /// Hardware GPU available (NVIDIA, AMD, etc.)
    HardwareGpu { name: String, backend: String },
    /// Software renderer (llvmpipe, lavapipe)
    SoftwareRenderer { name: String },
    /// No GPU available at all
    NotAvailable { reason: String },
}

pub struct GpuBackend {
    instance: Arc<Instance>,
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    status: GpuStatus,
}

impl GpuBackend {
    /// Try to create a GPU backend. Returns Ok even with software fallback.
    pub async fn new() -> Result<Self> {
        info!("Initializing GPU backend...");
        
        // First, run diagnostics
        Self::diagnose_gpu_environment();
        
        // Try to find the best adapter
        match Self::find_best_adapter().await {
            Ok((adapter, status)) => {
                let adapter_info = adapter.get_info();
                info!("✓ Selected GPU Adapter:");
                info!("  Name: {}", adapter_info.name);
                info!("  Vendor: 0x{:04x}", adapter_info.vendor);
                info!("  Device: 0x{:04x}", adapter_info.device);
                info!("  Backend: {:?}", adapter_info.backend);
                info!("  Device Type: {:?}", adapter_info.device_type);
                info!("  Driver: {}", adapter_info.driver);
                info!("  Driver Info: {}", adapter_info.driver_info);
                
                // Warn based on status
                match &status {
                    GpuStatus::SoftwareRenderer { name } => {
                        warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                        warn!("⚠️  Using software renderer: {}", name);
                        warn!("    ML training/inference will be VERY SLOW.");
                        warn!("");
                        warn!("    This is common with NVIDIA datacenter drivers (580.x)");
                        warn!("    which may not include Vulkan support.");
                        warn!("");
                        warn!("    Options:");
                        warn!("    1. Install nvidia-driver with Vulkan: apt install nvidia-driver-550");
                        warn!("    2. Use CUDA directly (not WebGPU) for compute");
                        warn!("    3. Continue with CPU-only ML (current fallback)");
                        warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    }
                    GpuStatus::HardwareGpu { name, backend } => {
                        info!("✓ Hardware GPU active: {} via {}", name, backend);
                    }
                    _ => {}
                }
                
                // Request device and queue
                info!("Requesting GPU device...");
                let (device, queue) = adapter
                    .request_device(
                        &DeviceDescriptor {
                            label: Some("WebGPU Host Device"),
                            required_features: Features::empty(),
                            required_limits: Self::get_compute_limits(&adapter),
                        },
                        None,
                    )
                    .await?;
                
                info!("✓ GPU device created");
                
                let instance = Instance::new(InstanceDescriptor {
                    backends: Backends::all(),
                    ..Default::default()
                });
                
                Ok(Self {
                    instance: Arc::new(instance),
                    adapter: Arc::new(adapter),
                    device: Arc::new(device),
                    queue: Arc::new(queue),
                    status,
                })
            }
            Err(e) => {
                Err(anyhow::anyhow!("Failed to initialize GPU: {}", e))
            }
        }
    }

    /// Diagnose the GPU environment and log findings
    fn diagnose_gpu_environment() {
        info!("GPU Environment Diagnostics:");
        
        // Check NVIDIA driver
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,driver_version,compute_cap")
            .arg("--format=csv,noheader")
            .output()
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                info!("  NVIDIA GPU: {}", info.trim());
            }
        }
        
        // Check environment variables
        if let Ok(icd) = std::env::var("VK_ICD_FILENAMES") {
            info!("  VK_ICD_FILENAMES: {}", icd);
        }
        if let Ok(driver) = std::env::var("VK_DRIVER_FILES") {
            info!("  VK_DRIVER_FILES: {}", driver);
        }
        
        // Check ICD file existence
        let icd_paths = [
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/etc/vulkan/icd.d/nvidia_icd.json",
        ];
        for path in &icd_paths {
            if std::path::Path::new(path).exists() {
                info!("  Found ICD: {}", path);
                // Try to read and show library path
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Some(lib_line) = content.lines().find(|l| l.contains("library_path")) {
                        info!("    {}", lib_line.trim());
                    }
                }
            }
        }
        
        // Check Vulkan loader
        let vulkan_libs = [
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib/libvulkan.so.1",
        ];
        for lib in &vulkan_libs {
            if std::path::Path::new(lib).exists() {
                info!("  Vulkan loader: {}", lib);
                break;
            }
        }
        
        // Check NVIDIA Vulkan library
        let nvidia_vulkan_libs = [
            "/usr/lib/x86_64-linux-gnu/libnvidia-vulkan-producer.so",
            "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        ];
        for lib in &nvidia_vulkan_libs {
            if std::path::Path::new(lib).exists() {
                info!("  NVIDIA Vulkan lib: {}", lib);
            }
        }
    }

    /// Find the best GPU adapter
    async fn find_best_adapter() -> Result<(Adapter, GpuStatus)> {
        info!("Searching for GPU adapters...");
        
        // Try Vulkan first (best for compute on Linux)
        let instance_vulkan = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            flags: InstanceFlags::empty(), // Don't require validation
            ..Default::default()
        });
        
        let vulkan_adapters: Vec<_> = instance_vulkan.enumerate_adapters(Backends::VULKAN).collect();
        info!("  Vulkan adapters found: {}", vulkan_adapters.len());
        
        for (i, adapter) in vulkan_adapters.iter().enumerate() {
            let info = adapter.get_info();
            info!("    [{}] {} ({:?})", i, info.name, info.device_type);
        }
        
        // Look for hardware GPU in Vulkan adapters
        for adapter in vulkan_adapters {
            let info = adapter.get_info();
            if info.device_type == DeviceType::DiscreteGpu || 
               info.device_type == DeviceType::IntegratedGpu {
                let status = GpuStatus::HardwareGpu {
                    name: info.name.clone(),
                    backend: format!("{:?}", info.backend),
                };
                return Ok((adapter, status));
            }
        }
        
        // Try all backends
        info!("  Trying all backends...");
        let instance_all = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::empty(),
            ..Default::default()
        });
        
        // First try to get a hardware GPU
        if let Some(adapter) = instance_all
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            let info = adapter.get_info();
            info!("    Found: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
            
            if info.device_type == DeviceType::DiscreteGpu ||
               info.device_type == DeviceType::IntegratedGpu {
                let status = GpuStatus::HardwareGpu {
                    name: info.name.clone(),
                    backend: format!("{:?}", info.backend),
                };
                return Ok((adapter, status));
            }
            
            // It's a software renderer but still usable
            if info.device_type == DeviceType::Cpu {
                let status = GpuStatus::SoftwareRenderer {
                    name: info.name.clone(),
                };
                return Ok((adapter, status));
            }
            
            // Virtual or other type
            let status = GpuStatus::SoftwareRenderer {
                name: info.name.clone(),
            };
            return Ok((adapter, status));
        }
        
        // Force fallback adapter (software renderer)
        info!("  Forcing fallback adapter...");
        if let Some(adapter) = instance_all
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            })
            .await
        {
            let info = adapter.get_info();
            let status = GpuStatus::SoftwareRenderer {
                name: info.name.clone(),
            };
            return Ok((adapter, status));
        }
        
        Err(anyhow::anyhow!("No GPU adapter found (not even software renderer)"))
    }

    /// Get appropriate limits for compute workloads
    fn get_compute_limits(adapter: &Adapter) -> Limits {
        let supported = adapter.limits();
        let mut limits = Limits::downlevel_defaults();
        
        limits.max_buffer_size = supported.max_buffer_size.min(256 * 1024 * 1024); // 256MB for safety
        limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size.min(128 * 1024 * 1024);
        limits.max_compute_workgroup_size_x = supported.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = supported.max_compute_workgroup_size_y;
        limits.max_compute_workgroup_size_z = supported.max_compute_workgroup_size_z;
        limits.max_compute_invocations_per_workgroup = supported.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroups_per_dimension = supported.max_compute_workgroups_per_dimension;
        
        limits
    }
    
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }

    pub fn is_hardware_gpu(&self) -> bool {
        matches!(self.status, GpuStatus::HardwareGpu { .. })
    }

    pub fn status(&self) -> &GpuStatus {
        &self.status
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    
    pub fn create_buffer(&self, size: u64, usage: BufferUsages, mapped_at_creation: bool) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some("WebGPU Host Buffer"),
            size,
            usage,
            mapped_at_creation,
        })
    }
    
    pub fn create_shader_module(&self, code: &str) -> ShaderModule {
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("WebGPU Host Shader"),
            source: ShaderSource::Wgsl(code.into()),
        })
    }
    
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }
    
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
    
    pub fn submit(&self, command_buffer: CommandBuffer) {
        self.queue.submit(Some(command_buffer));
    }
}

/// Check NVIDIA Vulkan setup and return diagnostic info
pub fn check_nvidia_vulkan_setup() -> Result<(), String> {
    use std::process::Command;
    
    // Check nvidia-smi
    let nvidia_smi = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version")
        .arg("--format=csv,noheader")
        .output();
    
    match nvidia_smi {
        Ok(output) if output.status.success() => {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            info!("NVIDIA GPU: {}", gpu_info.trim());
        }
        Ok(output) => {
            return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        Err(e) => {
            return Err(format!("nvidia-smi not found: {}", e));
        }
    }
    
    // Check ICD files
    let icd_paths = [
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "/etc/vulkan/icd.d/nvidia_icd.json",
    ];
    
    for path in &icd_paths {
        if std::path::Path::new(path).exists() {
            info!("Found NVIDIA Vulkan ICD: {}", path);
            return Ok(());
        }
    }
    
    warn!("NVIDIA Vulkan ICD not found");
    Ok(())
}
