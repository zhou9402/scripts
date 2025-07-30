#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 全局变量存储 Green Context 状态
static CUgreenCtx g_green_ctx = nullptr;
static CUcontext g_context = nullptr;
static CUcontext g_default_context = nullptr;
static bool g_initialized = false;

// ==================== TMA Helper Functions ====================
__device__ __forceinline__ void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta; \n" :: );
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster; \n" :: );
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" :: "r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("{\n\t"
                 ".reg .pred       P1; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
                 "@P1 bra DONE; \n\t"
                 "bra     LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}" :: "r"(mbar_int_ptr), "r"(phase), "r"(0x989680));
    phase ^= 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" :: "r"(num_bytes), "r"(mbar_int_ptr));
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = true) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes,
                                             bool evict_first = true) {
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n"
                 :: "l"(gmem_ptr), "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

// ==================== TMA Load Kernel ====================
// TMA Grid-stride Load Kernel with 32KB chunks
__global__ void tma_load_kernel(float4* src_data, float4* dst_data, int total_elements) {
    // 每次copy 32KB = 8192个float = 2048个float4
    const int elements_per_chunk = 2048;  // 32KB / 16 bytes per float4
    const int bytes_per_chunk = elements_per_chunk * 16;  // 32KB
    
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    
    // TMA barrier for load
    __shared__ uint64_t tma_mbarrier;
    uint32_t tma_phase = 0;

    if (threadIdx.x == 0) {
        mbarrier_init(&tma_mbarrier, 1);
        fence_view_async_shared();
        fence_barrier_init();
    }
    __syncthreads();

    // Grid-stride loop: 每个block轮流处理32KB数据块
    for (int chunk_start = blockIdx.x * elements_per_chunk; 
         chunk_start < total_elements; 
         chunk_start += gridDim.x * elements_per_chunk) {
        
        // 计算当前chunk的实际大小
        int chunk_end = min(chunk_start + elements_per_chunk, total_elements);
        int elements_to_process = chunk_end - chunk_start;
        int bytes_to_process = elements_to_process * 16;  // float4 = 16 bytes
        
        // 跳过空的chunk
        if (elements_to_process <= 0) {
            break;
        }

        // --- TMA Load: Global -> Shared ---
        if (threadIdx.x == 0)
        {
            // 计算当前chunk的源地址和目标地址
            const float4* chunk_src = src_data + chunk_start;
            float4* chunk_dst = dst_data + chunk_start;
            
            // TMA加载：从global memory的当前chunk位置加载到shared memory
            tma_load_1d(smem_buffer, chunk_src, &tma_mbarrier, bytes_to_process);
            mbarrier_arrive_and_expect_tx(&tma_mbarrier, bytes_to_process);
            
            // 等待TMA加载完成
            mbarrier_wait(&tma_mbarrier, tma_phase);
            
            // --- TMA Store: Shared -> Global ---
            // TMA存储：从shared memory存储到global memory的当前chunk位置
            tma_store_1d(smem_buffer, chunk_dst, bytes_to_process);
            tma_store_wait();
        }
        
        // 等待当前chunk的TMA操作完成再进行下一个chunk
        __syncthreads();
    }
}

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

// 检查GPU是否支持TMA (Hopper架构)
bool check_tma_support() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return prop.major >= 9;  // Hopper架构 (compute_90+)
}

// TMA Load Kernel接口 - 需要Hopper架构
void run_tma_load_kernel(torch::Tensor src_tensor) {
    // 基本检查
    TORCH_CHECK(src_tensor.device().is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(src_tensor.dtype() == torch::kFloat32, "Tensor must be float32");
    TORCH_CHECK(src_tensor.is_contiguous(), "Tensor must be contiguous");
    
    int tensor_elements = src_tensor.numel();
    TORCH_CHECK(tensor_elements >= 8, "Tensor too small, need at least 8 elements");
    
    // 获取源数据指针
    float4* src_data = reinterpret_cast<float4*>(src_tensor.data_ptr<float>());
    
    // 创建目标tensor
    torch::Tensor dst_tensor = torch::empty_like(src_tensor);
    float4* dst_data = reinterpret_cast<float4*>(dst_tensor.data_ptr<float>());
    
    // TMA kernel配置 - 32KB chunks, grid-stride loop
    int num_float4 = tensor_elements / 4;
    const int elements_per_chunk = 2048;  // 32KB / 16 bytes per float4
    
    // 计算需要的总chunk数量
    int total_chunks = (num_float4 + elements_per_chunk - 1) / elements_per_chunk;
    
    // 设置block数量：使用grid-stride pattern
    int threads_per_block = 1024;  // TMA需要足够的线程来占用SM
    
    // 设置shared memory大小 - 32KB per block
    size_t smem_size = elements_per_chunk * 16;  // 32KB
    
    // 设置kernel属性
    cudaFuncSetAttribute(tma_load_kernel, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size);
    
    // 调用TMA kernel
    tma_load_kernel<<<total_chunks, threads_per_block, smem_size>>>(src_data, dst_data, num_float4);
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
    
    m.def("run_tma_load_kernel", &run_tma_load_kernel, 
          "Run TMA load kernel with grid-stride loop (32KB chunks, requires Hopper)",
          py::arg("src_tensor"));
} 