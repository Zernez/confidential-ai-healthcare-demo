/**
 * GPU Backend using wgpu
 * 
 * Provides actual GPU access through wgpu library.
 * Optimized for headless server environments (like Azure VMs with H100).
 */

use anyhow::Result;
use wgpu::*;
use std::sync::Arc;
use log::{info, warn, debug, error};

pub struct GpuBackend {
    instance: Arc<Instance>,
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl GpuBackend {
    pub async fn new() -> Result<Self> {
        info!("Creating GPU instance...");
        
        // Try multiple backend configurations for headless servers
        let adapter = Self::find_best_adapter().await?;
        
        let adapter_info = adapter.get_info();
        info!("✓ Selected GPU Adapter:");
        info!("  Name: {}", adapter_info.name);
        info!("  Vendor: 0x{:04x}", adapter_info.vendor);
        info!("  Device: 0x{:04x}", adapter_info.device);
        info!("  Backend: {:?}", adapter_info.backend);
        info!("  Device Type: {:?}", adapter_info.device_type);
        info!("  Driver: {}", adapter_info.driver);
        info!("  Driver Info: {}", adapter_info.driver_info);
        
        // Warn if using software renderer
        if adapter_info.device_type == DeviceType::Cpu {
            warn!("⚠️  Using CPU/software renderer - GPU compute will be slow!");
            warn!("    This usually means the NVIDIA Vulkan driver is not properly configured.");
            warn!("    Try setting: export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json");
        }
        
        // Request device and queue with appropriate limits for compute
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
        
        // Create instance for storage (we already have the adapter)
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });
        
        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Find the best GPU adapter, preferring discrete GPUs over integrated/software
    async fn find_best_adapter() -> Result<Adapter> {
        // Strategy 1: Try Vulkan with explicit NVIDIA preference
        info!("Searching for GPU adapters...");
        
        // Check environment variable for ICD override
        if let Ok(icd) = std::env::var("VK_ICD_FILENAMES") {
            info!("  VK_ICD_FILENAMES={}", icd);
        } else {
            debug!("  VK_ICD_FILENAMES not set");
        }
        
        // Create instance with Vulkan backend (primary for compute on Linux)
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            dx12_shader_compiler: Dx12Compiler::default(),
            flags: InstanceFlags::default(),
            gles_minor_version: Gles3MinorVersion::default(),
        });
        
        // Enumerate all adapters
        let adapters = instance.enumerate_adapters(Backends::VULKAN);
        
        info!("Found {} Vulkan adapter(s):", adapters.len());
        for (i, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            info!("  [{}] {} ({:?}, {:?})", 
                i, info.name, info.device_type, info.backend);
            debug!("      Vendor: 0x{:04x}, Device: 0x{:04x}", info.vendor, info.device);
            debug!("      Driver: {} - {}", info.driver, info.driver_info);
        }
        
        // Priority order for adapter selection:
        // 1. Discrete GPU (NVIDIA, AMD)
        // 2. Integrated GPU
        // 3. Virtual GPU
        // 4. CPU (software fallback - avoid if possible)
        
        // First, look for NVIDIA discrete GPU specifically
        for adapter in &adapters {
            let info = adapter.get_info();
            // NVIDIA vendor ID is 0x10de
            if info.vendor == 0x10de && info.device_type == DeviceType::DiscreteGpu {
                info!("✓ Found NVIDIA discrete GPU: {}", info.name);
                // Need to get a new adapter since we can't move from &adapters
                return instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Failed to request NVIDIA adapter"));
            }
        }
        
        // Look for any discrete GPU
        for adapter in &adapters {
            let info = adapter.get_info();
            if info.device_type == DeviceType::DiscreteGpu {
                info!("✓ Found discrete GPU: {}", info.name);
                return instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Failed to request discrete GPU adapter"));
            }
        }
        
        // Look for integrated GPU (better than software)
        for adapter in &adapters {
            let info = adapter.get_info();
            if info.device_type == DeviceType::IntegratedGpu {
                info!("✓ Found integrated GPU: {}", info.name);
                return instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Failed to request integrated GPU adapter"));
            }
        }
        
        // Strategy 2: Try with all backends (including potential CUDA backend in future)
        warn!("No discrete/integrated GPU found with Vulkan, trying all backends...");
        
        let instance_all = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        if let Some(adapter) = instance_all
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            let info = adapter.get_info();
            if info.device_type != DeviceType::Cpu {
                info!("✓ Found GPU via alternative backend: {} ({:?})", info.name, info.backend);
                return Ok(adapter);
            }
        }
        
        // Last resort: accept software renderer but warn heavily
        error!("⚠️  No hardware GPU found! Falling back to software renderer.");
        error!("    Performance will be extremely poor for ML workloads.");
        error!("");
        error!("    To fix this on Azure/Linux with NVIDIA GPU:");
        error!("    1. Ensure NVIDIA driver is installed: nvidia-smi");
        error!("    2. Install Vulkan ICD: sudo apt install nvidia-vulkan-icd");
        error!("    3. Set ICD path: export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json");
        error!("    4. For headless: export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json");
        error!("");
        
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,  // Accept software fallback
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found at all"))
    }

    /// Get appropriate limits for compute workloads
    fn get_compute_limits(adapter: &Adapter) -> Limits {
        let supported = adapter.limits();
        
        // Use downlevel limits as base, then increase for compute
        let mut limits = Limits::downlevel_defaults();
        
        // Increase buffer sizes for ML workloads
        limits.max_buffer_size = supported.max_buffer_size.min(1024 * 1024 * 1024); // 1GB max
        limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size.min(512 * 1024 * 1024);
        limits.max_compute_workgroup_size_x = supported.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = supported.max_compute_workgroup_size_y;
        limits.max_compute_workgroup_size_z = supported.max_compute_workgroup_size_z;
        limits.max_compute_invocations_per_workgroup = supported.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroups_per_dimension = supported.max_compute_workgroups_per_dimension;
        
        debug!("Compute limits:");
        debug!("  max_buffer_size: {} MB", limits.max_buffer_size / 1024 / 1024);
        debug!("  max_storage_buffer_binding_size: {} MB", limits.max_storage_buffer_binding_size / 1024 / 1024);
        debug!("  max_compute_workgroup_size: {}x{}x{}", 
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z);
        
        limits
    }
    
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }

    pub fn is_hardware_gpu(&self) -> bool {
        let info = self.adapter.get_info();
        matches!(info.device_type, DeviceType::DiscreteGpu | DeviceType::IntegratedGpu)
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

/// Check if NVIDIA Vulkan driver is properly configured
pub fn check_nvidia_vulkan_setup() -> Result<(), String> {
    use std::process::Command;
    
    // Check if nvidia-smi works
    let nvidia_smi = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version")
        .arg("--format=csv,noheader")
        .output();
    
    match nvidia_smi {
        Ok(output) if output.status.success() => {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            info!("NVIDIA GPU detected: {}", gpu_info.trim());
        }
        Ok(output) => {
            return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        Err(e) => {
            return Err(format!("nvidia-smi not found: {}. NVIDIA driver may not be installed.", e));
        }
    }
    
    // Check Vulkan ICD files
    let icd_paths = [
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "/etc/vulkan/icd.d/nvidia_icd.json",
        "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
    ];
    
    let mut found_icd = false;
    for path in &icd_paths {
        if std::path::Path::new(path).exists() {
            info!("Found NVIDIA Vulkan ICD: {}", path);
            found_icd = true;
            break;
        }
    }
    
    if !found_icd {
        warn!("NVIDIA Vulkan ICD not found in standard locations");
        warn!("Try: sudo apt install nvidia-vulkan-icd");
        warn!("Or set VK_ICD_FILENAMES to point to your nvidia_icd.json");
    }
    
    Ok(())
}
