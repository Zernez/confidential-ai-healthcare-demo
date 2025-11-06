cmake_minimum_required(VERSION 3.20)
project(wasmwebgpu-ml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ═══════════════════════════════════════════════
# Build Configuration
# ═══════════════════════════════════════════════

option(BUILD_WASM "Build for WebAssembly with WASI" ON)
option(BUILD_NATIVE "Build native binary for testing" OFF)

# Include external dependencies
include_directories(${CMAKE_SOURCE_DIR}/external)

# ═══════════════════════════════════════════════
# WASM-Specific Configuration
# ═══════════════════════════════════════════════

if(BUILD_WASM)
    # WASI target
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    
    # Compiler flags for WASM
    add_compile_options(
        -O3                          # Optimize for speed
        -flto                        # Link-time optimization
        -fno-exceptions              # No exceptions (reduce size)
        --target=wasm32-wasip2       # WASI Preview 2
    )
    
    # Linker flags for WASM
    add_link_options(
        -Wl,--strip-all              # Strip debug symbols
        -Wl,--gc-sections            # Remove unused sections
        -Wl,--lto-O3                 # LTO optimization
        -mexec-model=reactor         # WASI reactor model
    )
    
    message(STATUS "Building for WASM target (wasm32-wasip2)")
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
    src/wasi_webgpu_wrapper.hpp
)

# ═══════════════════════════════════════════════
# Executable Target
# ═══════════════════════════════════════════════

add_executable(wasmwebgpu-ml-benchmark ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(wasmwebgpu-ml-benchmark
    # Math library
    m
)

# ═══════════════════════════════════════════════
# Install Rules
# ═══════════════════════════════════════════════

install(TARGETS wasmwebgpu-ml-benchmark
    RUNTIME DESTINATION bin
)

# Copy shaders
install(DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
    DESTINATION share/wasmwebgpu-ml
)

# ═══════════════════════════════════════════════
# Print Configuration
# ═══════════════════════════════════════════════

message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  wasmwebgpu-ml Configuration")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  WASM Build: ${BUILD_WASM}")
message(STATUS "  Native Build: ${BUILD_NATIVE}")
if(BUILD_WASM)
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "  Target: wasm32-wasip2")
endif()
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
