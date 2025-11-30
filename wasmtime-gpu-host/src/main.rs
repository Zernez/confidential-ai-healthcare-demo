//! Wasmtime GPU Host Runtime
//!
//! Runs WASM modules that use the wasi:gpu interface.
//! Automatically selects between CUDA and WebGPU backends.
//! Supports TEE attestation (AMD SEV-SNP + NVIDIA CC).

mod backend;
mod host;
mod tee_host;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tracing::{error, info, warn};
use wasmtime::*;
use wasmtime_wasi::preview1::{self, WasiP1Ctx};
use wasmtime_wasi::WasiCtxBuilder;

use backend::GpuBackend;
use host::GpuState;
use tee_host::{TeeHost, AsTeeHost};

#[cfg(feature = "cuda")]
use backend::cuda::CudaBackend;
use backend::webgpu::WebGpuBackend;

/// Wasmtime GPU Host - Run WASM modules with GPU acceleration
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the WASM module
    #[arg(required = true)]
    wasm_path: PathBuf,
    
    /// Force a specific GPU backend
    #[arg(short, long, value_parser = ["cuda", "webgpu", "auto"])]
    backend: Option<String>,
    
    /// Working directory for the WASM module
    #[arg(short = 'd', long)]
    workdir: Option<PathBuf>,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Arguments to pass to the WASM module
    #[arg(last = true)]
    wasm_args: Vec<String>,
}

/// Combined state for WASI + GPU + TEE
struct HostState {
    wasi: WasiP1Ctx,
    gpu: GpuState,
    tee: TeeHost,
}

impl AsTeeHost for HostState {
    fn as_tee_host(&self) -> &TeeHost {
        &self.tee
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level))
        )
        .init();
    
    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║       Wasmtime GPU Host Runtime                          ║");
    info!("║       wasi:gpu + TEE attestation (CUDA + WebGPU)         ║");
    info!("╚══════════════════════════════════════════════════════════╝");
    
    // Select GPU backend
    let backend: Box<dyn GpuBackend> = select_backend(&args.backend)?;
    
    info!("[Runtime] Using backend: {}", backend.device_info().backend);
    info!("[Runtime] Device: {}", backend.device_info().name);
    
    // Initialize TEE Host
    let tee_host = TeeHost::new();
    let tee_type = tee_host.detect_tee_type();
    info!("[Runtime] TEE type: {}", tee_type);
    
    // Create engine
    let mut config = Config::new();
    config.wasm_component_model(false); // Use core WASM, not component model yet
    config.async_support(false);
    
    let engine = Engine::new(&config)?;
    
    // Load WASM module
    info!("[Runtime] Loading WASM: {}", args.wasm_path.display());
    
    let wasm_bytes = std::fs::read(&args.wasm_path)
        .with_context(|| format!("Failed to read WASM file: {}", args.wasm_path.display()))?;
    
    let module = Module::new(&engine, &wasm_bytes)
        .with_context(|| "Failed to compile WASM module")?;
    
    info!("[Runtime] Module compiled successfully");
    
    // Create linker
    let mut linker = Linker::new(&engine);
    
    // Add WASI functions
    preview1::add_to_linker_sync(&mut linker, |state: &mut HostState| &mut state.wasi)?;
    
    // Add wasi:gpu functions
    host::add_to_linker(&mut linker, |state: &mut HostState| &mut state.gpu)?;
    
    // Add attestation functions
    tee_host.register_functions(&mut linker)?;
    
    // Build WASI context
    let workdir = args.workdir.unwrap_or_else(|| std::env::current_dir().unwrap());
    
    let mut wasi_builder = WasiCtxBuilder::new();
    wasi_builder
        .inherit_stdio()
        .inherit_env()
        .args(&args.wasm_args)
        .preopened_dir(&workdir, ".", wasmtime_wasi::DirPerms::all(), wasmtime_wasi::FilePerms::all())?
        .preopened_dir(&workdir, &workdir.to_string_lossy(), wasmtime_wasi::DirPerms::all(), wasmtime_wasi::FilePerms::all())?;
    
    // Add data directory if it exists
    let data_dir = workdir.join("data");
    if data_dir.exists() {
        wasi_builder.preopened_dir(&data_dir, "data", wasmtime_wasi::DirPerms::all(), wasmtime_wasi::FilePerms::all())?;
    }
    
    let wasi = wasi_builder.build_p1();
    
    // Create store with state
    let state = HostState {
        wasi,
        gpu: GpuState::new(backend),
        tee: tee_host,
    };
    
    let mut store = Store::new(&engine, state);
    
    // Instantiate module
    info!("[Runtime] Instantiating module...");
    
    let instance = linker.instantiate(&mut store, &module)
        .with_context(|| "Failed to instantiate WASM module")?;
    
    // Find and call _start (WASI entry point)
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .with_context(|| "Module has no _start function")?;
    
    info!("[Runtime] Executing WASM module...");
    info!("────────────────────────────────────────────────────────────");
    
    match start.call(&mut store, ()) {
        Ok(()) => {
            info!("────────────────────────────────────────────────────────────");
            info!("[Runtime] WASM execution completed successfully");
        }
        Err(e) => {
            error!("────────────────────────────────────────────────────────────");
            error!("[Runtime] WASM execution failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

/// Select the best available GPU backend
fn select_backend(preference: &Option<String>) -> Result<Box<dyn GpuBackend>> {
    let pref = preference.as_deref().unwrap_or("auto");
    
    match pref {
        "cuda" => {
            info!("[Backend] Requested CUDA backend");
            #[cfg(feature = "cuda")]
            {
                match CudaBackend::new() {
                    Ok(backend) => return Ok(Box::new(backend)),
                    Err(e) => {
                        error!("[Backend] CUDA init failed: {:?}", e);
                        anyhow::bail!("CUDA backend requested but not available");
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA feature not compiled in");
            }
        }
        "webgpu" => {
            info!("[Backend] Requested WebGPU backend");
            match WebGpuBackend::new_sync() {
                Ok(backend) => return Ok(Box::new(backend)),
                Err(e) => {
                    error!("[Backend] WebGPU init failed: {:?}", e);
                    anyhow::bail!("WebGPU backend requested but not available");
                }
            }
        }
        "auto" | _ => {
            info!("[Backend] Auto-detecting best backend...");
            
            // Try CUDA first (preferred for H100)
            #[cfg(feature = "cuda")]
            {
                match CudaBackend::new() {
                    Ok(backend) => {
                        info!("[Backend] CUDA backend available");
                        return Ok(Box::new(backend));
                    }
                    Err(e) => {
                        warn!("[Backend] CUDA not available: {:?}", e);
                    }
                }
            }
            
            // Fall back to WebGPU
            match WebGpuBackend::new_sync() {
                Ok(backend) => {
                    info!("[Backend] WebGPU backend available");
                    return Ok(Box::new(backend));
                }
                Err(e) => {
                    error!("[Backend] WebGPU not available: {:?}", e);
                }
            }
            
            anyhow::bail!("No GPU backend available");
        }
    }
}
