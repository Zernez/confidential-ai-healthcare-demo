/**
 * Custom Wasmtime Host with wasi:webgpu Support
 * 
 * This implements a WASM runtime that provides GPU access through wasi:webgpu API
 * to WASM guests. It uses wgpu for actual GPU operations.
 */

use anyhow::{Context, Result};
use wasmtime::*;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};
use std::path::PathBuf;
use log::{info, error};

mod webgpu_host;
mod gpu_backend;

use webgpu_host::WebGpuHost;

/// Command line arguments
#[derive(Debug)]
struct Args {
    wasm_file: PathBuf,
    dirs: Vec<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        anyhow::bail!("Usage: {} <wasm_file> [--dir <directory>]...", args[0]);
    }
    
    let wasm_file = PathBuf::from(&args[1]);
    let mut dirs = Vec::new();
    
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--dir" && i + 1 < args.len() {
            dirs.push(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }
    
    Ok(Args { wasm_file, dirs })
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default()
        .default_filter_or("info"))
        .init();
    
    info!("╔════════════════════════════════════════════════╗");
    info!("║  Wasmtime with wasi:webgpu Support (Beta)     ║");
    info!("╚════════════════════════════════════════════════╝");
    
    // Parse arguments
    let args = parse_args()?;
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
    
    // Initialize GPU backend
    let gpu_backend = pollster::block_on(gpu_backend::GpuBackend::new())
        .context("Failed to initialize GPU")?;
    
    info!("✓ GPU backend initialized");
    info!("  GPU: {}", gpu_backend.adapter_info());
    
    // Create WebGPU host implementation
    let webgpu_host = WebGpuHost::new(gpu_backend);
    
    // Register wasi:webgpu functions
    webgpu_host.register_functions(&mut linker)?;
    
    info!("✓ wasi:webgpu functions registered");
    
    // Create store with host state
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_args()?;
    
    // Add preopened directories
    let mut wasi_builder = wasi;
    for dir in &args.dirs {
        info!("Adding directory: {:?}", dir);
        wasi_builder = wasi_builder.preopened_dir(
            wasmtime_wasi::sync::Dir::open_ambient_dir(dir, wasmtime_wasi::sync::ambient_authority())?,
            ".",
        )?;
    }
    
    let wasi_ctx = wasi_builder.build();
    
    let mut store = Store::new(
        &engine,
        HostState {
            wasi: wasi_ctx,
            webgpu: webgpu_host,
        },
    );
    
    // Load WASM module
    info!("Loading WASM module...");
    let module = Module::from_file(&engine, &args.wasm_file)
        .context("Failed to load WASM module")?;
    
    info!("✓ WASM module loaded");
    
    // Instantiate module
    info!("Instantiating module...");
    let instance = linker.instantiate(&mut store, &module)
        .context("Failed to instantiate module")?;
    
    // Find and call _start (WASI entry point)
    info!("Running WASM...");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .context("Failed to find _start function")?;
    
    start.call(&mut store, ())
        .context("WASM execution failed")?;
    
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("✓ WASM execution completed successfully");
    
    Ok(())
}

/// Host state containing WASI context and WebGPU implementation
struct HostState {
    wasi: WasiCtx,
    webgpu: WebGpuHost,
}
