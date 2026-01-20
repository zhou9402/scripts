#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

// 定义内存访问函数
__device__ __forceinline__ void st_na_global(int4 *ptr, const int4& value) {
    asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" 
                 :: "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) 
                 : "memory");
}

__device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
    int4 value;
    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];" 
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) 
                 : "l"(ptr) 
                 : "memory");
    return value;
}

// UNROLLED_WARP_COPY 宏定义
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
    do { \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
        int4 unrolled_values[(UNROLL_FACTOR)]; \
        auto __src = (SRC); \
        auto __dst = (DST); \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
        } \
        for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32) \
            ST_FUNC(__dst + __i, LD_FUNC(__src + __i)); \
    } while (0)

// 简单数据搬移版本 - 每个线程处理一个元素
__global__ void row_move_kernel_simple(float* src, float* dst, int N, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N * cols; i += total_threads) {
        dst[i] = src[i];
    }
}

// 向量化数据搬移版本 - 每个线程处理4个float
__global__ void row_move_kernel_vectorized(float* src, float* dst, int N, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 确保cols能被4整除
    int vec_cols = cols / 4;
    float4* vec_src = (float4*)src;
    float4* vec_dst = (float4*)dst;
    
    for (int i = idx; i < N * vec_cols; i += total_threads) {
        vec_dst[i] = vec_src[i];
    }
}

// 向量化数据搬移版本 - 每个block处理一行，block分配连续任务
__global__ void row_move_kernel_bypass_l1_vectorized(float* src, float* dst, int N, int cols) {
    int block_id = blockIdx.x;  // block ID
    int total_blocks = gridDim.x;
    int thread_id = threadIdx.x;  // thread ID within block
    int threads_per_block = blockDim.x;
    
    // 只使用第一个warp（前32个线程）
    if (thread_id >= 32) return;
    
    // 计算每个block分配的行数范围（连续分配）
    int rows_per_block = (N + total_blocks - 1) / total_blocks;  // 向上取整
    int start_row = block_id * rows_per_block;
    int end_row = min(start_row + rows_per_block, N);
    
    // 确保cols能被4整除，以便进行int4读写
    int vec_cols = cols / 4;
    int4* vec_src = (int4*)src;
    int4* vec_dst = (int4*)dst;
    
    // 每个block处理分配给它的连续行
    for (int row = start_row; row < end_row; row++) {
        // 当前block内只有第一个warp的线程协作搬移一行数据，使用UNROLLED_WARP_COPY
        UNROLLED_WARP_COPY(4, thread_id, vec_cols,
                          vec_dst + row * vec_cols,
                          vec_src + row * vec_cols,
                          ld_nc_global, st_na_global);
    }
}

// Initialize memory with specific pattern
__global__ void init_memory_pattern(int* data, size_t num_elements, int pattern) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        data[i] = pattern;
    }
}

// 性能测试函数（带内存初始化）
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, float* d_src, float* d_dst, int N, int cols, 
                      int grid_size, int block_size, const char* kernel_name) {
    const int num_warmup = 5;
    const int num_iterations = 100;
    
    const int SRC_PATTERN = 0xCAFEBABE;  // Source pattern
    const int DST_PATTERN = 0xBAADF00D;  // Destination pattern
    
    int* d_src_int = reinterpret_cast<int*>(d_src);
    int* d_dst_int = reinterpret_cast<int*>(d_dst);
    size_t num_elements = N * cols;
    
    int init_grid_size = (num_elements + 1023) / 1024;
    
    // 预热
    printf("Warming up %s with memory initialization...\n", kernel_name);
    for (int i = 0; i < num_warmup; i++) {
        init_memory_pattern<<<init_grid_size, 1024>>>(d_src_int, num_elements, SRC_PATTERN);
        init_memory_pattern<<<init_grid_size, 1024>>>(d_dst_int, num_elements, DST_PATTERN);
        cudaDeviceSynchronize();
        kernel<<<grid_size, block_size>>>(d_src, d_dst, N, cols);
        cudaDeviceSynchronize();
    }
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> times(num_iterations);
    
    printf("Running %d measurements with memory initialization (src=0xCAFEBABE, dst=0xBAADF00D)...\n", num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        // Initialize memory before each measurement
        init_memory_pattern<<<init_grid_size, 1024>>>(d_src_int, num_elements, SRC_PATTERN);
        init_memory_pattern<<<init_grid_size, 1024>>>(d_dst_int, num_elements, DST_PATTERN);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        kernel<<<grid_size, block_size>>>(d_src, d_dst, N, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&times[i], start, stop);
        if (i < 10 || i % 10 == 9) {
            printf("  Run %3d: %.3f ms\n", i + 1, times[i]);
        }
    }
    
    // Calculate statistics
    float min_time = times[0];
    float max_time = times[0];
    float total_time = 0.0f;
    
    for (int i = 0; i < num_iterations; i++) {
        total_time += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    
    float avg_time = total_time / num_iterations;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float diff = times[i] - avg_time;
        variance += diff * diff;
    }
    float std_dev = sqrt(variance / num_iterations);
    
    // Calculate bandwidth statistics
    size_t data_size = N * cols * sizeof(float);
    double avg_bandwidth = (data_size * 2 / (avg_time * 1e-3)) / 1e9; // GB/s (read + write)
    double min_bandwidth = (data_size * 2 / (max_time * 1e-3)) / 1e9;
    double max_bandwidth = (data_size * 2 / (min_time * 1e-3)) / 1e9;
    
    printf("\n=== %s Performance Summary ===\n", kernel_name);
    printf("Time (ms):\n");
    printf("  Min:    %.3f\n", min_time);
    printf("  Max:    %.3f\n", max_time);
    printf("  Avg:    %.3f ± %.3f\n", avg_time, std_dev);
    printf("Bandwidth (GB/s):\n");
    printf("  Min:    %.1f\n", min_bandwidth);
    printf("  Max:    %.1f\n", max_bandwidth);
    printf("  Avg:    %.1f\n", avg_bandwidth);
    printf("Coefficient of Variation: %.2f%%\n\n", (std_dev / avg_time) * 100.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_time;
}

// 验证结果正确性
bool verify_result(float* h_src, float* h_dst, int N, int cols) {
    for (int i = 0; i < N * cols; i++) {
        if (abs(h_src[i] - h_dst[i]) > 1e-6) {
            printf("Verification failed at index %d: expected %f, got %f\n", i, h_src[i], h_dst[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 10240;      // 行数
    const int cols = 4096;   // 列数，确保能被4整除
    const size_t size = N * cols * sizeof(float);
    
    printf("Matrix size: %d x %d\n", N, cols);
    printf("Total elements: %d\n", N * cols);
    printf("Memory size: %.2f MB\n", size / 1024.0 / 1024.0);
    
    // 分配主机内存
    std::vector<float> h_src(N * cols);
    std::vector<float> h_dst(N * cols);
    
    // 初始化数据
    for (int i = 0; i < N * cols; i++) {
        h_src[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    // 分配设备内存
    float *d_src, *d_dst;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_src, h_src.data(), size, cudaMemcpyHostToDevice);
    
    // 测试不同的内核
    printf("\nBenchmarking kernels:\n");
    printf("=============================================================================\n\n");
    
    // 简单版本
    benchmark_kernel(row_move_kernel_simple, d_src, d_dst, N, cols, 
                    (N * cols + 255) / 256, 256, "Simple kernel");
    
    // 验证简单版本结果
    cudaMemcpy(h_dst.data(), d_dst, size, cudaMemcpyDeviceToHost);
    if (verify_result(h_src.data(), h_dst.data(), N, cols)) {
        printf("✅ Simple kernel: PASSED\n\n");
    } else {
        printf("❌ Simple kernel: FAILED\n\n");
    }
    
    printf("=============================================================================\n\n");
    
    // 向量化版本
    benchmark_kernel(row_move_kernel_vectorized, d_src, d_dst, N, cols, 
                    (N * cols / 4 + 255) / 256, 256, "Vectorized kernel");
    
    // 验证向量化版本结果
    cudaMemcpy(h_dst.data(), d_dst, size, cudaMemcpyDeviceToHost);
    if (verify_result(h_src.data(), h_dst.data(), N, cols)) {
        printf("✅ Vectorized kernel: PASSED\n\n");
    } else {
        printf("❌ Vectorized kernel: FAILED\n\n");
    }
    
    printf("=============================================================================\n\n");
    
    // 绕过L1缓存的向量化版本
    benchmark_kernel(row_move_kernel_bypass_l1_vectorized, d_src, d_dst, N, cols, 
                    N, 256, "Bypass L1 vectorized kernel");
    
    // 验证绕过L1缓存版本结果
    cudaMemcpy(h_dst.data(), d_dst, size, cudaMemcpyDeviceToHost);
    if (verify_result(h_src.data(), h_dst.data(), N, cols)) {
        printf("✅ Bypass L1 vectorized kernel: PASSED\n\n");
    } else {
        printf("❌ Bypass L1 vectorized kernel: FAILED\n\n");
    }
    
    // 清理内存
    cudaFree(d_src);
    cudaFree(d_dst);
    
    printf("=============================================================================\n");
    printf("All tests completed!\n");
    
    return 0;
}