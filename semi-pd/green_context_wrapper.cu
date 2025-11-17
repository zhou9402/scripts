#include <torch/extension.h>
#include <cuda.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 全局变量存储 Green Context 状态
static CUcontext g_primary_context = nullptr;
static CUgreenCtx g_primary_partition_green_ctx = nullptr;
static CUgreenCtx g_remaining_partition_green_ctx = nullptr;
static CUstream g_primary_partition_stream = nullptr;
static CUstream g_remaining_partition_stream = nullptr;
static int g_primary_partition_sm_count = 0;
static int g_remaining_partition_sm_count = 0;
static bool g_initialized = false;

// 创建Green Context和Streams (一次性划分SM为primary和remaining)
std::tuple<uint64_t, uint64_t, int, int> create_green_context_and_streams(
    int intended_primary_sm_count,
    int primary_stream_priority = 0,
    int remaining_stream_priority = 0,
    int device_id = 0) {
    
    if (g_initialized) {
        throw std::runtime_error("Green Context already initialized. Call destroy first.");
    }
    
    // 初始化 CUDA Driver
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver");
    }
    
    // 获取设备
    CUdevice device;
    res = cuDeviceGet(&device, device_id);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get CUDA device");
    }
    
    // 获取primary context
    res = cuDevicePrimaryCtxRetain(&g_primary_context, device);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to retain primary context");
    }
    
    // 设置当前context
    res = cuCtxSetCurrent(g_primary_context);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to set current context");
    }
    
    // 1. 获取设备的SM资源
    CUdevResource device_resource;
    res = cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get device SM resource");
    }
    
    int total_sm_count = device_resource.sm.smCount;
    std::cout << "[Green Context] Total SM count: " << total_sm_count << std::endl;
    
    if (intended_primary_sm_count > total_sm_count) {
        throw std::runtime_error("Requested SM count exceeds available SMs");
    }
    
    // 2. 分割SM资源为primary和remaining partitions
    CUdevResource primary_partition_resource;
    CUdevResource remaining_partition_resource;
    unsigned int num_groups = 1;
    
    res = cuDevSmResourceSplitByCount(
        &primary_partition_resource,
        &num_groups,
        &device_resource,
        &remaining_partition_resource,
        0,  // flags
        intended_primary_sm_count
    );
    
    if (res != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to split SM resources");
    }
    
    g_primary_partition_sm_count = primary_partition_resource.sm.smCount;
    g_remaining_partition_sm_count = remaining_partition_resource.sm.smCount;
    
    std::cout << "[Green Context] Primary partition: " << g_primary_partition_sm_count << " SMs" << std::endl;
    std::cout << "[Green Context] Remaining partition: " << g_remaining_partition_sm_count << " SMs" << std::endl;
    
    // 3. 生成资源描述符
    CUdevResourceDesc primary_partition_desc;
    CUdevResourceDesc remaining_partition_desc;
    
    res = cuDevResourceGenerateDesc(&primary_partition_desc, &primary_partition_resource, 1);
    if (res != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to generate primary partition descriptor");
    }
    
    res = cuDevResourceGenerateDesc(&remaining_partition_desc, &remaining_partition_resource, 1);
    if (res != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to generate remaining partition descriptor");
    }
    
    // 4. 创建Green Contexts
    unsigned int green_ctx_flags = CU_GREEN_CTX_DEFAULT_STREAM;
    
    res = cuGreenCtxCreate(&g_primary_partition_green_ctx, primary_partition_desc, device, green_ctx_flags);
    if (res != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to create primary partition green context");
    }
    
    res = cuGreenCtxCreate(&g_remaining_partition_green_ctx, remaining_partition_desc, device, green_ctx_flags);
    if (res != CUDA_SUCCESS) {
        cuGreenCtxDestroy(g_primary_partition_green_ctx);
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to create remaining partition green context");
    }
    
    // 5. 创建Streams
    unsigned int stream_flags = CU_STREAM_NON_BLOCKING;
    
    res = cuGreenCtxStreamCreate(&g_primary_partition_stream, g_primary_partition_green_ctx, 
                                 stream_flags, primary_stream_priority);
    if (res != CUDA_SUCCESS) {
        cuGreenCtxDestroy(g_remaining_partition_green_ctx);
        cuGreenCtxDestroy(g_primary_partition_green_ctx);
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to create primary partition stream");
    }
    
    res = cuGreenCtxStreamCreate(&g_remaining_partition_stream, g_remaining_partition_green_ctx, 
                                 stream_flags, remaining_stream_priority);
    if (res != CUDA_SUCCESS) {
        cuStreamDestroy(g_primary_partition_stream);
        cuGreenCtxDestroy(g_remaining_partition_green_ctx);
        cuGreenCtxDestroy(g_primary_partition_green_ctx);
        cuDevicePrimaryCtxRelease(device);
        throw std::runtime_error("Failed to create remaining partition stream");
    }
    
    g_initialized = true;
    
    std::cout << "[Green Context] Created successfully!" << std::endl;
    std::cout << "  Primary stream: 0x" << std::hex << reinterpret_cast<uint64_t>(g_primary_partition_stream) << std::dec << std::endl;
    std::cout << "  Remaining stream: 0x" << std::hex << reinterpret_cast<uint64_t>(g_remaining_partition_stream) << std::dec << std::endl;
    
    // 返回stream handles和SM counts
    return std::make_tuple(
        reinterpret_cast<uint64_t>(g_primary_partition_stream),
        reinterpret_cast<uint64_t>(g_remaining_partition_stream),
        g_primary_partition_sm_count,
        g_remaining_partition_sm_count
    );
}

// 销毁Green Context
void destroy_green_context() {
    if (!g_initialized) {
        return;
    }
    
    std::cout << "[Green Context] Destroying..." << std::endl;
    
    if (g_remaining_partition_stream) {
        cuStreamDestroy(g_remaining_partition_stream);
        g_remaining_partition_stream = nullptr;
    }
    
    if (g_primary_partition_stream) {
        cuStreamDestroy(g_primary_partition_stream);
        g_primary_partition_stream = nullptr;
    }
    
    if (g_remaining_partition_green_ctx) {
        cuGreenCtxDestroy(g_remaining_partition_green_ctx);
        g_remaining_partition_green_ctx = nullptr;
    }
    
    if (g_primary_partition_green_ctx) {
        cuGreenCtxDestroy(g_primary_partition_green_ctx);
        g_primary_partition_green_ctx = nullptr;
    }
    
    if (g_primary_context) {
        CUdevice device;
        cuCtxGetDevice(&device);
        cuDevicePrimaryCtxRelease(device);
        g_primary_context = nullptr;
    }
    
    g_initialized = false;
    
    std::cout << "[Green Context] Destroyed successfully!" << std::endl;
}

// 获取SM分配信息
std::tuple<int, int> get_sm_counts() {
    if (!g_initialized) {
        throw std::runtime_error("Green Context not initialized");
    }
    return std::make_tuple(g_primary_partition_sm_count, g_remaining_partition_sm_count);
}

// 检查是否已初始化
bool is_initialized() {
    return g_initialized;
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Green Context Library with Primary/Remaining Partition Support";
    
    m.def("create_green_context_and_streams", &create_green_context_and_streams,
          "Create Green Context with primary and remaining partitions, returns (primary_stream, remaining_stream, primary_sms, remaining_sms)",
          py::arg("intended_primary_sm_count"),
          py::arg("primary_stream_priority") = 0,
          py::arg("remaining_stream_priority") = 0,
          py::arg("device_id") = 0);
    
    m.def("destroy_green_context", &destroy_green_context,
          "Destroy Green Context and free all resources");
    
    m.def("get_sm_counts", &get_sm_counts,
          "Get SM counts for primary and remaining partitions, returns (primary_sms, remaining_sms)");
    
    m.def("is_initialized", &is_initialized,
          "Check if Green Context is initialized");
}
