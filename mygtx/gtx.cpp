//
// Simple Green Context Library with Python Bindings
//

#include "gtx.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

// Simple wrapper class
class SimpleGreenContext {
private:
    CUdevice device;
    CUcontext primary_context;
    CUgreenCtx primary_partition_green_context;
    CUgreenCtx remaining_partition_green_context;
    CUstream primary_partition_stream;
    CUstream remaining_partition_stream;
    int primary_partition_sm_count;
    int remaining_partition_sm_count;
    bool initialized;

public:
    SimpleGreenContext(int device_id = 0) : initialized(false) {
        // Initialize CUDA
        CUresult result = cuInit(0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA");
        }
        
        // Get device
        result = cuDeviceGet(&device, device_id);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device");
        }
        
        // Retain primary context
        result = cuDevicePrimaryCtxRetain(&primary_context, device);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to retain primary context");
        }
        
        initialized = true;
    }
    
    ~SimpleGreenContext() {
        if (initialized) {
            destroy_streams();
            cuDevicePrimaryCtxRelease(device);
        }
    }
    
    std::pair<uint64_t, uint64_t> create_green_context_and_streams(
        int intended_primary_partition_sm_count,
        int primary_stream_priority = 0,
        int remaining_stream_priority = 0) {
        
        if (!initialized) {
            throw std::runtime_error("Green context not initialized");
        }
        
        // Set current context
        CUresult result = cuCtxSetCurrent(primary_context);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to set current context");
        }
        
        // Create Green Context and streams
        result = create_green_context_and_stream_1t1g(
            0, // device_id
            intended_primary_partition_sm_count,
            primary_stream_priority,
            remaining_stream_priority,
            primary_context,
            primary_partition_green_context,
            remaining_partition_green_context,
            primary_partition_stream,
            remaining_partition_stream,
            primary_partition_sm_count,
            remaining_partition_sm_count);
        
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create green context and streams");
        }
        
        // Return stream handles as integers
        return std::make_pair(
            reinterpret_cast<uint64_t>(primary_partition_stream),
            reinterpret_cast<uint64_t>(remaining_partition_stream)
        );
    }
    
    void destroy_streams() {
        if (initialized && primary_partition_stream && remaining_partition_stream) {
            cuCtxSetCurrent(primary_context);
            destroy_green_context_and_stream_1t1g(
                device,
                primary_context,
                primary_partition_green_context,
                remaining_partition_green_context,
                primary_partition_stream,
                remaining_partition_stream);
            
            primary_partition_stream = nullptr;
            remaining_partition_stream = nullptr;
        }
    }
    
    std::pair<int, int> get_sm_counts() {
        return std::make_pair(primary_partition_sm_count, remaining_partition_sm_count);
    }
};

// Standalone function for convenience
std::pair<uint64_t, uint64_t> create_green_context_and_streams_simple(
    int intended_primary_partition_sm_count,
    int primary_stream_priority = 0,
    int remaining_stream_priority = 0,
    int device_id = 0) {
    
    auto manager = std::make_shared<SimpleGreenContext>(device_id);
    return manager->create_green_context_and_streams(
        intended_primary_partition_sm_count,
        primary_stream_priority,
        remaining_stream_priority);
}

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(green_context_simple, m) {
    m.doc() = "Simple Green Context Library for Python";
    
    py::class_<SimpleGreenContext>(m, "GreenContextManager")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("create_green_context_and_streams", 
             &SimpleGreenContext::create_green_context_and_streams,
             py::arg("intended_primary_partition_sm_count"),
             py::arg("primary_stream_priority") = 0,
             py::arg("remaining_stream_priority") = 0,
             "Create Green Context and return two stream handles")
        .def("destroy_streams", 
             &SimpleGreenContext::destroy_streams,
             "Destroy Green Context and streams")
        .def("get_sm_counts", 
             &SimpleGreenContext::get_sm_counts,
             "Get SM counts for primary and remaining partitions");
    
    m.def("create_green_context_and_streams", 
          &create_green_context_and_streams_simple,
          py::arg("intended_primary_partition_sm_count"),
          py::arg("primary_stream_priority") = 0,
          py::arg("remaining_stream_priority") = 0,
          py::arg("device_id") = 0,
          "Create Green Context and return two stream handles");
    
    m.attr("__version__") = "1.0.0";
} 