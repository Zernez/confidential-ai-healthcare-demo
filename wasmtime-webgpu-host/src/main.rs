/**
 * Custom Wasmtime Host with wasi:webgpu Support
 * 
 * This implements a WASM runtime that provides GPU access through wasi:webgpu API
 * to WASM guests. It uses wgpu for actual GPU operations.
 * 
 * Optimized for headless server environments (Azure VMs with NVIDIA H100).
 */

use anyhow::{Context, Result};
use wasmtime::*;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};
use std::path::PathBuf;
use log::{info, warn, error};

mod webgpu_host;
mod gpu_backend;
mod tee_host;

use webgpu_host::WebGpuHost;
use tee_host::{TeeHost, AsTeeHost};

/// Command line arguments
#[derive(Debug)]
struct Args {
    wasm_file: PathBuf,
    dirs: Vec<PathBuf>,
    check_gpu: bool,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <wasm_file> [--dir <directory>]... [--check-gpu]", args[0]);
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --dir <directory>   Add a directory accessible to WASM");
        eprintln!("  --check-gpu         Check GPU/Vulkan setup and exit");
        eprintln!("");
        eprintln!("Environment variables:");
        eprintln!("  RUST_LOG=debug      Enable debug logging");
        eprintln!("  VK_ICD_FILENAMES    Override Vulkan ICD path");
        eprintln!("  VK_DRIVER_FILES     Alternative Vulkan driver path");
        std::process::exit(1);
    }
    
    let wasm_file = PathBuf::from(&args[1]);
    let mut dirs = Vec::new();
    let mut check_gpu = false;
    
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" if i + 1 < args.len() => {
                dirs.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--check-gpu" => {
                check_gpu = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    
    Ok(Args { wasm_file, dirs, check_gpu })
}

/// Setup environment for headless GPU compute
fn setup_gpu_environment() {
    // Check if we're in a headless environment
    if std::env::var("DISPLAY").is_err() {
        info!("Running in headless mode (no DISPLAY)");
        
        // For headless NVIDIA Vulkan, we may need to set the ICD path
        if std::env::var("VK_ICD_FILENAMES").is_err() {
            // Try common NVIDIA ICD locations
            let icd_paths = [
                "/usr/share/vulkan/icd.d/nvidia_icd.json",
                "/etc/vulkan/icd.d/nvidia_icd.json", 
                "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
            ];
            
            for path in &icd_paths {
                if std::path::Path::new(path).exists() {
                    info!("Setting VK_ICD_FILENAMES={}", path);
                    std::env::set_var("VK_ICD_FILENAMES", path);
                    break;
                }
            }
        }
        
        // Also try VK_DRIVER_FILES for newer Vulkan loaders
        if std::env::var("VK_DRIVER_FILES").is_err() {
            if let Ok(icd) = std::env::var("VK_ICD_FILENAMES") {
                std::env::set_var("VK_DRIVER_FILES", &icd);
            }
        }
    }
    
    // Log current Vulkan environment
    if let Ok(icd) = std::env::var("VK_ICD_FILENAMES") {
        info!("VK_ICD_FILENAMES={}", icd);
    }
    if let Ok(driver) = std::env::var("VK_DRIVER_FILES") {
        info!("VK_DRIVER_FILES={}", driver);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default()
        .default_filter_or("info"))
        .init();
    
    info!("╔════════════════════════════════════════════════╗");
    info!("║  Wasmtime with TEE Attestation + WebGPU       ║");
    info!("║  • wasi:webgpu (GPU compute)                  ║");
    info!("║  • wasmtime:attestation (VM + GPU)            ║");
    info!("╚════════════════════════════════════════════════╝");
    
    // Parse arguments
    let args = parse_args()?;
    
    // Setup GPU environment for headless operation
    setup_gpu_environment();
    
    // If --check-gpu flag, just check and exit
    if args.check_gpu {
        info!("Checking GPU/Vulkan setup...");
        
        // Check NVIDIA driver
        if let Err(e) = gpu_backend::check_nvidia_vulkan_setup() {
            error!("NVIDIA check failed: {}", e);
        }
        
        // Try to initialize GPU
        match gpu_backend::GpuBackend::new().await {
            Ok(backend) => {
                info!("✓ GPU initialization successful!");
                info!("  Adapter: {}", backend.adapter_info());
                info!("  Hardware GPU: {}", backend.is_hardware_gpu());
                
                if !backend.is_hardware_gpu() {
                    warn!("⚠️  Using software renderer - see suggestions above to fix");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                error!("✗ GPU initialization failed: {}", e);
                std::process::exit(1);
            }
        }
        
        return Ok(());
    }
    
    info!("Loading WASM: {:?}", args.wasm_file);
    
    // Create Wasmtime engine
    let mut config = Config::new();
    config.wasm_backtrace_details(WasmBacktraceDetails::Enable);
    config.async_support(true);
    let engine = Engine::new(&config)?;
    
    // Create linker
    let mut linker = Linker::new(&engine);
    
    // Add WASI support
    wasmtime_wasi::add_to_linker(&mut linker, |state: &mut HostState| &mut state.wasi)?;
    
    info!("Initializing GPU backend...");
    
    // Check NVIDIA setup first (non-fatal)
    if let Err(e) = gpu_backend::check_nvidia_vulkan_setup() {
        warn!("NVIDIA Vulkan check: {}", e);
    }
    
    // Initialize GPU backend
    let gpu_backend = gpu_backend::GpuBackend::new().await
        .context("Failed to initialize GPU")?;
    
    info!("GPU backend initialized");
    info!("  GPU: {}", gpu_backend.adapter_info());
    
    if !gpu_backend.is_hardware_gpu() {
        warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        warn!("⚠️  WARNING: Using software GPU renderer!");
        warn!("    ML training/inference will be VERY SLOW.");
        warn!("    Run with --check-gpu to diagnose the issue.");
        warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
    
    // Create WebGPU host implementation
    let webgpu_host = WebGpuHost::new(gpu_backend);
    
    // Register wasi:webgpu functions
    webgpu_host.register_functions(&mut linker)?;
    
    info!("✓ wasi:webgpu functions registered");
    
    // Create TEE attestation host
    info!("Initializing TEE attestation...");
    let tee_host = TeeHost::new();
    
    // Register wasmtime:attestation functions
    tee_host.register_functions(&mut linker)?;
    
    info!("✓ wasmtime:attestation functions registered");
    
    // Create store with host state
    let mut wasi_builder = WasiCtxBuilder::new();
    wasi_builder.inherit_stdio();
    wasi_builder.inherit_args()?;
    
    // Add preopened directories
    for dir in &args.dirs {
        info!("Adding directory: {:?}", dir);
        
        // Mount each directory with its name
        let mount_name = dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".");
        
        info!("  Mounting as: {:?}", mount_name);
        
        wasi_builder.preopened_dir(
            wasmtime_wasi::sync::Dir::open_ambient_dir(dir, wasmtime_wasi::sync::ambient_authority())?,
            mount_name,
        )?;
    }
    let wasi_ctx = wasi_builder.build();
    
    let mut store = Store::new(
        &engine,
        HostState {
            wasi: wasi_ctx,
            webgpu: webgpu_host,
            tee: tee_host,
        },
    );
    
    // Load WASM module
    info!("Loading WASM module...");
    let module = Module::from_file(&engine, &args.wasm_file)
        .context("Failed to load WASM module")?;
    
    info!("WASM module loaded");
    
    // Instantiate module
    info!("Instantiating module...");
    let instance = linker.instantiate_async(&mut store, &module).await
        .context("Failed to instantiate module")?;
    
    // Find and call _start (WASI entry point)
    info!("Running WASM...");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .context("Failed to find _start function")?;
    
    start.call_async(&mut store, ()).await
        .context("WASM execution failed")?;
    
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("WASM execution completed successfully");
    
    Ok(())
}

/// Host state containing WASI context, WebGPU, and TEE attestation
struct HostState {
    wasi: WasiCtx,
    webgpu: WebGpuHost,
    tee: TeeHost,
}

/// Implement AsTeeHost trait for HostState
impl AsTeeHost for HostState {
    fn as_tee_host(&self) -> &TeeHost {
        &self.tee
    }
}
