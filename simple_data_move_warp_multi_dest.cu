#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 使用CUDA内置函数绕过L1缓存的版本 - SM内warp轮流处理行 - 向量化版本
__global__ void row_copy_kernel_bypass_l1_builtin_vectorized(float* dst, float* src, int N, int cols) {
    int sm_id = blockIdx.x;
    int total_sms = gridDim.x;
    int warp_id = threadIdx.x / 32;  // 当前warp在block中的ID
    int lane_id = threadIdx.x % 32;  // 当前线程在warp中的ID
    int warps_per_block = blockDim.x / 32;  // 每个block的warp数量
    
    // 每个SM处理的行数
    int rows_per_sm = (N + total_sms - 1) / total_sms;
    int start_row = sm_id * rows_per_sm;
    int end_row = min(start_row + rows_per_sm, N);
    
    // 向量化处理，每次处理4个float
    int vec_cols = cols / 4;  // 向量化后的列数
    float4* vec_src = (float4*)src;
    float4* vec_dst = (float4*)dst;
    
    // SM内的warp轮流处理分配给该SM的行
    for (int row = start_row + warp_id; row < end_row; row += warps_per_block) {
        // warp内的线程协作拷贝一行数据，使用向量化访问
        #pragma unroll 4
        for (int i = lane_id; i < vec_cols; i += 32) {
            // 绕过L1缓存读取4个float
            float4 data = __ldcg(&vec_src[row * vec_cols + i]);
            // 绕过L1缓存写入4个float
            __stcg(&vec_dst[row * vec_cols + i], data);
        }
        
        // 处理剩余的元素（如果cols不是4的倍数）
        int remaining_start = vec_cols * 4;
        for (int i = remaining_start + lane_id; i < cols; i += 32) {
            float data = __ldcg(&src[row * cols + i]);
            __stcg(&dst[row * cols + i], data);
        }
    }
}

// 4目的地数据移动kernel - 使用warp shuffle避免共享内存
__global__ void row_copy_kernel_4_destinations(float* dst0, float* dst1, float* dst2, float* dst3, 
                                               float* src, int N, int cols) {
    int sm_id = blockIdx.x;
    int total_sms = gridDim.x;
    int warp_id = threadIdx.x / 32;  // 当前warp在block中的ID
    int lane_id = threadIdx.x % 32;  // 当前线程在warp中的ID
    int warps_per_block = blockDim.x / 32;  // 每个block的warp数量
    
    // 每个SM处理的行数
    int rows_per_sm = (N + total_sms - 1) / total_sms;
    int start_row = sm_id * rows_per_sm;
    int end_row = min(start_row + rows_per_sm, N);
    
    // 向量化处理，每次处理4个float
    int vec_cols = cols / 4;  // 向量化后的列数
    float4* vec_src = (float4*)src;
    
    // 目的地指针数组
    float* dst_ptrs[4] = {dst0, dst1, dst2, dst3};
    float4* vec_dst_ptrs[4] = {(float4*)dst0, (float4*)dst1, (float4*)dst2, (float4*)dst3};
    
    // 确定当前warp负责的目的地
    int dest_id = warp_id % 4;  // 0, 1, 2, 3 循环
    int warp_group = warp_id / 4;  // 第几组4个warp
    int max_warp_groups = warps_per_block / 4;
    
    // 直接获取当前warp对应的目的地指针
    float4* current_vec_dst = vec_dst_ptrs[dest_id];
    float* current_dst = dst_ptrs[dest_id];
    
    // SM内的warp组轮流处理分配给该SM的行
    for (int row = start_row + warp_group; row < end_row; row += max_warp_groups) {
        // 所有4个warp协作处理同一行
        // 每个warp处理不同的列段，然后通过shuffle共享数据
        
        // 计算每个warp负责的列范围
        int cols_per_warp = (vec_cols + 3) / 4;  // 向上取整
        int my_start_col = dest_id * cols_per_warp;
        int my_end_col = min(my_start_col + cols_per_warp, vec_cols);
        
        // 第一阶段：每个warp加载自己负责的数据段
        for (int base_col = 0; base_col < vec_cols; base_col += cols_per_warp * 4) {
            // 每个warp在自己的列段内工作
            for (int local_col = my_start_col; local_col < my_end_col; local_col += 32) {
                int global_col = base_col + local_col + lane_id;
                
                if (global_col < vec_cols) {
                    // 加载数据
                    float4 data = __ldcg(&vec_src[row * vec_cols + global_col]);
                    
                    // 写入到所有4个目的地
                    __stcg(&vec_dst_ptrs[0][row * vec_cols + global_col], data);
                    __stcg(&vec_dst_ptrs[1][row * vec_cols + global_col], data);
                    __stcg(&vec_dst_ptrs[2][row * vec_cols + global_col], data);
                    __stcg(&vec_dst_ptrs[3][row * vec_cols + global_col], data);
                }
            }
        }
        
        // 处理剩余的标量元素
        int remaining_start = vec_cols * 4;
        int remaining_count = cols - remaining_start;
        
        if (remaining_count > 0) {
            // 类似处理剩余元素
            int scalar_cols_per_warp = (remaining_count + 3) / 4;
            int my_scalar_start = dest_id * scalar_cols_per_warp;
            int my_scalar_end = min(my_scalar_start + scalar_cols_per_warp, remaining_count);
            
            for (int i = my_scalar_start + lane_id; i < my_scalar_end; i += 32) {
                if (i < remaining_count) {
                    float data = __ldcg(&src[row * cols + remaining_start + i]);
                    
                    // 写入到所有4个目的地
                    __stcg(&dst_ptrs[0][row * cols + remaining_start + i], data);
                    __stcg(&dst_ptrs[1][row * cols + remaining_start + i], data);
                    __stcg(&dst_ptrs[2][row * cols + remaining_start + i], data);
                    __stcg(&dst_ptrs[3][row * cols + remaining_start + i], data);
                }
            }
        }
    }
}

// L2缓存flush kernel
__global__ void flush_l2_cache(char* dummy_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 访问足够大的内存区域来flush L2缓存
    for (size_t i = idx; i < size; i += stride) {
        dummy_data[i] = (char)(i & 0xFF);
    }
}

int main(int argc, char** argv) {
    // 默认参数
    int N = 40960;           // 行数
    int cols = 1792;        // 确保是4的倍数以便向量化
    int num_sms = 80;       // SM数量
    int threads_per_block = 1024;  // 恢复到1024线程
    
    // 解析命令行参数
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_sms = atoi(argv[2]);
    
    // 确保cols是4的倍数，便于向量化
    cols = (cols + 3) / 4 * 4;
    
    // 确保每个block的warp数量是4的倍数
    int warps_per_block = threads_per_block / 32;
    if (warps_per_block % 4 != 0) {
        warps_per_block = (warps_per_block / 4) * 4;
        threads_per_block = warps_per_block * 32;
    }
    
    printf("4-Destination Data Movement Test (No Shared Memory)\n");
    printf("Matrix: %d x %d floats\n", N, cols);
    printf("Threads per block: %d (%d warps, %d warp groups)\n", 
           threads_per_block, warps_per_block, warps_per_block/4);
    
    // 分配内存
    size_t size = N * cols * sizeof(float);
    float *src, *dst0, *dst1, *dst2, *dst3;
    CUDA_CHECK(cudaMalloc(&src, size));
    CUDA_CHECK(cudaMalloc(&dst0, size));
    CUDA_CHECK(cudaMalloc(&dst1, size));
    CUDA_CHECK(cudaMalloc(&dst2, size));
    CUDA_CHECK(cudaMalloc(&dst3, size));
    
    // 分配用于flush L2缓存的dummy数据
    // L2缓存大小通常是几MB到几十MB，我们分配足够大的内存
    size_t l2_flush_size = 128 * 1024 * 1024; // 128MB，足以flush大部分GPU的L2缓存
    char *dummy_data;
    CUDA_CHECK(cudaMalloc(&dummy_data, l2_flush_size));
    
    // 初始化数据
    CUDA_CHECK(cudaMemset(src, 1, size));
    CUDA_CHECK(cudaMemset(dst0, 0, size));
    CUDA_CHECK(cudaMemset(dst1, 0, size));
    CUDA_CHECK(cudaMemset(dst2, 0, size));
    CUDA_CHECK(cudaMemset(dst3, 0, size));
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Flush L2缓存
    printf("Flushing L2 cache...\n");
    flush_l2_cache<<<num_sms, threads_per_block>>>(dummy_data, l2_flush_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 执行4目的地拷贝kernel，不使用共享内存
    CUDA_CHECK(cudaEventRecord(start));
    row_copy_kernel_4_destinations<<<num_sms, threads_per_block>>>(
        dst0, dst1, dst2, dst3, src, N, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    // 总数据量：1次读取 + 4次写入 = 5倍数据量
    double bandwidth = (size * 5.0 / (time_ms / 1000.0)) / 1e9;
    
    printf("Time: %.2f ms\n", time_ms);
    printf("Bandwidth: %.1f GB/s (1 read + 4 writes)\n", bandwidth);
    printf("Effective copy rate: %.1f GB/s per destination\n", bandwidth / 4.0);
    
    // 清理
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst0));
    CUDA_CHECK(cudaFree(dst1));
    CUDA_CHECK(cudaFree(dst2));
    CUDA_CHECK(cudaFree(dst3));
    CUDA_CHECK(cudaFree(dummy_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
} 