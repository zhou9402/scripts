#!/bin/bash

set -e  # Exit on error

echo "=== Building Green Context Simple Library ==="

# CUDA path detection
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "Error: CUDA_HOME not set and /usr/local/cuda not found"
        exit 1
    fi
fi

echo "Using CUDA_HOME: $CUDA_HOME"

# Check for required dependencies
python3 -c "import pybind11" 2>/dev/null || {
    echo "Error: pybind11 not found. Please install: pip install pybind11"
    exit 1
}

# Clean previous builds
echo "Cleaning previous builds..."
rm -f green_context_simple*.so
rm -rf build/
rm -rf *.egg-info/

# Get pybind11 includes
PYBIND11_INCLUDES=$(python3 -m pybind11 --includes)

# Compiler flags
CXX_FLAGS="-O3 -Wall -shared -std=c++17 -fPIC"
CUDA_FLAGS="-I${CUDA_HOME}/include"
CUDA_LIBS="-L${CUDA_HOME}/lib64 -lcuda -lcudart"

# Python extension suffix
PYTHON_EXT_SUFFIX=$(python3-config --extension-suffix)

echo "Compiling green_context_simple..."

# Compile the extension
g++ ${CXX_FLAGS} ${PYBIND11_INCLUDES} ${CUDA_FLAGS} \
    green_context_simple.cpp gtx.cpp \
    ${CUDA_LIBS} \
    -o green_context_simple${PYTHON_EXT_SUFFIX}

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Generated: green_context_simple${PYTHON_EXT_SUFFIX}"
    echo ""
    echo "Usage example:"
    echo "  import green_context_simple as gc"
    echo "  manager = gc.GreenContextManager()"
    echo "  primary_stream, remaining_stream = manager.create_green_context_and_streams(120)"
    echo "  print(f'Primary stream: 0x{primary_stream:x}, Remaining stream: 0x{remaining_stream:x}')"
else
    echo "❌ Build failed!"
    exit 1
fi 