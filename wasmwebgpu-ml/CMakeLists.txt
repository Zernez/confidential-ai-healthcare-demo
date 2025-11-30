cmake_minimum_required(VERSION 3.20)
project(wasmwebgpu-ml VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ═══════════════════════════════════════════════
# Build Configuration
# ═══════════════════════════════════════════════

option(BUILD_WASM "Build for WebAssembly with WASI" ON)
option(BUILD_NATIVE "Build native binary for testing" OFF)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/external)

# ═══════════════════════════════════════════════
# WASM-Specific Configuration (wasi:gpu)
# ═══════════════════════════════════════════════

if(BUILD_WASM)
    # WASI target
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    
    # Compiler flags for WASM component model
    add_compile_options(
        -O3                          # Optimize for speed
        -flto                        # Link-time optimization
        -fno-exceptions              # No exceptions (reduce size)
        --target=wasm32-wasi         # WASI target
    )
    
    # Linker flags for WASM
    add_link_options(
        -Wl,--strip-all              # Strip debug symbols
        -Wl,--gc-sections            # Remove unused sections
        -Wl,--lto-O3                 # LTO optimization
        # Allow undefined symbols for wasi:gpu imports
        -Wl,--allow-undefined
    )
    
    message(STATUS "Building for WASM target with wasi:gpu interface")
endif()

# ═══════════════════════════════════════════════
# Source Files
# ═══════════════════════════════════════════════

set(SOURCES
    src/main.cpp
    src/dataset.cpp
    src/random_forest.cpp
    src/gpu_executor.cpp
)

set(HEADERS
    src/dataset.hpp
    src/random_forest.hpp
    src/gpu_executor.hpp
    src/wasi_gpu.h
    src/attestation.hpp
)

# ═══════════════════════════════════════════════
# Executable Target
# ═══════════════════════════════════════════════

add_executable(wasmwebgpu-ml-benchmark ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(wasmwebgpu-ml-benchmark
    m  # Math library
)

# ═══════════════════════════════════════════════
# Install Rules
# ═══════════════════════════════════════════════

install(TARGETS wasmwebgpu-ml-benchmark
    RUNTIME DESTINATION bin
)

# Copy WIT files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/wit
    DESTINATION share/wasmwebgpu-ml
)

# ═══════════════════════════════════════════════
# Print Configuration
# ═══════════════════════════════════════════════

message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  wasmwebgpu-ml Configuration (v${PROJECT_VERSION})")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  WASM Build: ${BUILD_WASM}")
message(STATUS "  GPU Interface: wasi:gpu (host-provided)")
if(BUILD_WASM)
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "  Target: wasm32-wasi")
endif()
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
