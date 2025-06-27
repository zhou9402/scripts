#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
    
// 使用UNROLLED_WARP_COPY的向量化数据搬移版本 - 并行执行
__global__ void row_move_kernel_unrolled_warp_copy(float* src, float* dst, int N, int cols) {
    int sm_id = blockIdx.x;
    int total_sms = gridDim.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int active_warps_per_block = 32;
    if (warp_id >= active_warps_per_block) {
        return;
    }
    
    // 计算每个SM分配的行数范围
    int rows_per_sm = (N + total_sms - 1) / total_sms;
    int start_row = sm_id * rows_per_sm;
    int end_row = min(start_row + rows_per_sm, N);
    
    // 确保cols能被4整除，以便进行int4读写
    int vec_cols = cols / 4;
    int4* vec_src = (int4*)src;
    int4* vec_dst = (int4*)dst;
    
    // SM内所有warp并行处理行
    for (int row = start_row + warp_id; row < end_row; row += active_warps_per_block) {
        // 使用UNROLLED_WARP_COPY宏来搬移一行数据
        int4* row_src = vec_src + row * vec_cols;
        int4* row_dst = vec_dst + row * vec_cols;
        
        // 使用展开因子5的UNROLLED_WARP_COPY
        UNROLLED_WARP_COPY(5, lane_id, vec_cols, row_dst, row_src, ld_nc_global, st_na_global);
    }
}

// 添加flush L2缓存的kernel
__global__ void flush_l2_cache(float* dummy_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 访问足够大的内存区域来flush L2缓存
    for (size_t i = idx; i < size / sizeof(float); i += stride) {
        dummy_data[i] = 0.0f;
    }
}

int main(int argc, char** argv) {
    // 默认参数
    int N = 40960;           // 行数
    int cols = 8192;        // 确保能被4整除以支持float4读写
    int num_sms = 1;       // SM数量
    int threads_per_block = 1024;  // 使用1024个线程
    
    // 解析命令行参数
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_sms = atoi(argv[2]);
    
    // 确保cols对齐到4的倍数
    cols = (cols + 3) & ~3;  // 4字节对齐
    
    printf("Vectorized Data Move Test - Bypass L1 Cache (int4)\n");
    printf("Matrix: %d x %d floats\n", N, cols);
    
    // 分配内存
    size_t size = N * cols * sizeof(float);
    float *src, *dst;
    CUDA_CHECK(cudaMalloc(&src, size));
    CUDA_CHECK(cudaMalloc(&dst, size));
    
    // 初始化数据
    CUDA_CHECK(cudaMemset(src, 1, size));
    CUDA_CHECK(cudaMemset(dst, 0, size));
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 多次测量参数
    const int num_warmup = 5;    // 预热次数
    const int num_runs = 20;     // 测量次数
    float times[num_runs];
    
    printf("Warming up...\n");
    // 预热运行
    for (int i = 0; i < num_warmup; i++) {
        flush_l2_cache<<<1, 1024>>>(src, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        row_move_kernel_unrolled_warp_copy<<<num_sms, threads_per_block>>>(src, dst, N, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    printf("Running %d measurements...\n", num_runs);
    // 正式测量
    for (int i = 0; i < num_runs; i++) {
        // 每次测量前flush L2缓存
        flush_l2_cache<<<1, 1024>>>(src, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        row_move_kernel_unrolled_warp_copy<<<num_sms, threads_per_block>>>(src, dst, N, cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        printf("Run %2d: %.3f ms\n", i+1, times[i]);
    }
    
    // 计算统计信息
    float min_time = times[0];
    float max_time = times[0];
    float total_time = 0.0f;
    
    for (int i = 0; i < num_runs; i++) {
        total_time += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    
    float avg_time = total_time / num_runs;
    
    // 计算标准差
    float variance = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        float diff = times[i] - avg_time;
        variance += diff * diff;
    }
    float std_dev = sqrt(variance / num_runs);
    
    // 数据搬移的带宽计算：实际搬移的数据量
    double avg_bandwidth = (size / (avg_time / 1000.0)) / 1e9 * 2;
    double min_bandwidth = (size / (max_time / 1000.0)) / 1e9 * 2;  // 最大时间对应最小带宽
    double max_bandwidth = (size / (min_time / 1000.0)) / 1e9 * 2;  // 最小时间对应最大带宽
    
    printf("\n=== Performance Summary ===\n");
    printf("Time (ms):\n");
    printf("  Min:    %.3f\n", min_time);
    printf("  Max:    %.3f\n", max_time);
    printf("  Avg:    %.3f ± %.3f\n", avg_time, std_dev);
    printf("Bandwidth (GB/s):\n");
    printf("  Min:    %.1f\n", min_bandwidth);
    printf("  Max:    %.1f\n", max_bandwidth);
    printf("  Avg:    %.1f\n", avg_bandwidth);
    printf("Coefficient of Variation: %.2f%%\n", (std_dev / avg_time) * 100.0f);
    
    // 清理
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
} 
