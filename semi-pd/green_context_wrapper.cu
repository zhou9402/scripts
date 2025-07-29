#include <torch/extension.h>
#include <cuda.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 全局变量存储 Green Context 状态
static CUgreenCtx g_green_ctx = nullptr;
static CUcontext g_context = nullptr;
static CUcontext g_default_context = nullptr;
static bool g_initialized = false;

bool create_green_context(int sm_count = 8) {
    if (g_initialized) {
        return true; // 已经初始化
    }
    
    // 保存当前的默认context
    CUresult res = cuCtxGetCurrent(&g_default_context);
    if (res != CUDA_SUCCESS) {
        std::cerr << "无法获取当前默认context" << std::endl;
        return false;
    }
    
    CUdevice device;
    CUdevResource dev_resource = {};
    CUdevResource sm_resources[2] = {{}, {}};
    CUdevResourceDesc desc = nullptr;
    unsigned int flags = CU_GREEN_CTX_DEFAULT_STREAM;
    unsigned int split_count = 1;
    unsigned int min_sm_count = sm_count;
    
    // 初始化 CUDA Driver
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA初始化失败" << std::endl;
        return false;
    }
    
    // 获取设备
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "获取设备失败" << std::endl;
        return false;
    }
    
    // 获取设备的 SM 资源
    res = cuDeviceGetDevResource(device, &dev_resource, CU_DEV_RESOURCE_TYPE_SM);
    if (res != CUDA_SUCCESS) {
        std::cerr << "获取设备资源失败" << std::endl;
        return false;
    }
    
    // 分割 SM 资源
    res = cuDevSmResourceSplitByCount(&sm_resources[0], &split_count, 
                                      &dev_resource, &sm_resources[1], 
                                      0, min_sm_count);
    if (res != CUDA_SUCCESS) {
        std::cerr << "分割 SM 资源失败" << std::endl;
        return false;
    }
    
    // 生成资源描述符
    res = cuDevResourceGenerateDesc(&desc, &sm_resources[0], 1);
    if (res != CUDA_SUCCESS) {
        std::cerr << "生成资源描述符失败" << std::endl;
        return false;
    }
    
    // 创建 Green Context
    res = cuGreenCtxCreate(&g_green_ctx, desc, device, flags);
    if (res != CUDA_SUCCESS) {
        std::cerr << "创建Green Context失败" << std::endl;
        return false;
    }
    
    // 转换为普通 context
    res = cuCtxFromGreenCtx(&g_context, g_green_ctx);
    if (res != CUDA_SUCCESS) {
        std::cerr << "转换 Green Context 失败" << std::endl;
        return false;
    }
    
    g_initialized = true;
    return true;
}

void destroy_green_context() {
    if (g_initialized && g_green_ctx) {
        // 先切换回默认context
        if (g_default_context) {
            cuCtxSetCurrent(g_default_context);
        }
        cuGreenCtxDestroy(g_green_ctx);
        g_green_ctx = nullptr;
        g_context = nullptr;
        g_default_context = nullptr;
        g_initialized = false;
    }
}

bool switch_to_green_context() {
    if (!g_initialized || !g_context) {
        std::cerr << "Green Context未初始化，请先调用create_green_context" << std::endl;
        return false;
    }
    
    CUresult res = cuCtxSetCurrent(g_context);
    if (res != CUDA_SUCCESS) {
        std::cerr << "切换到Green Context失败" << std::endl;
        return false;
    }
    
    return true;
}

bool switch_to_default_context() {
    if (!g_default_context) {
        std::cerr << "默认Context不可用" << std::endl;
        return false;
    }
    
    CUresult res = cuCtxSetCurrent(g_default_context);
    if (res != CUDA_SUCCESS) {
        std::cerr << "切换到默认Context失败" << std::endl;
        return false;
    }
    
    return true;
}

bool is_green_context_active() {
    return g_initialized;
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_green_context", &create_green_context, 
          "Create Green Context with limited SM count", 
          py::arg("sm_count") = 8);
    
    m.def("destroy_green_context", &destroy_green_context, 
          "Destroy Green Context");
    
    m.def("switch_to_green_context", &switch_to_green_context, 
          "Switch to Green Context");
    
    m.def("switch_to_default_context", &switch_to_default_context, 
          "Switch to default Context");
    
    m.def("is_green_context_active", &is_green_context_active, 
          "Check if Green Context is active");
} 