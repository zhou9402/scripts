#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", __FILE__, __LINE__, error, cudaGetErrorName(error), cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)



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

__device__ __forceinline__ void tma_store_fence() {
    asm volatile ("fence.proxy.async.shared::cta;");
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


// A proper TMA bulk copy kernel for Hopper - Grid-stride loop with 32KB per iteration
__global__ void tma_bulk_copy_kernel(
    void *__restrict__ vdst,
    const void *__restrict__ vsrc,
    int N)
{
    // 每次copy 32KB = 8192个float
    const int elements_per_chunk = 8192;  // 32KB / 4 bytes per float
    const int bytes_per_chunk = elements_per_chunk * sizeof(float);  // 32KB
    
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto *dst = reinterpret_cast<float *>(vdst);
    const auto *src = reinterpret_cast<const float *>(vsrc);
    
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
         chunk_start < N; 
         chunk_start += gridDim.x * elements_per_chunk) {
        
        // 计算当前chunk的实际大小
        int chunk_end = min(chunk_start + elements_per_chunk, N);
        int elements_to_process = chunk_end - chunk_start;
        int bytes_to_process = elements_to_process * sizeof(float);
        
        // 跳过空的chunk
        if (elements_to_process <= 0) {
            break;
        }

        // --- TMA Load: Global -> Shared ---
        if (threadIdx.x == 0)
        {
            // 计算当前chunk的源地址和目标地址
            const float* chunk_src = src + chunk_start;
            float* chunk_dst = dst + chunk_start;
            
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

int main(int argc, char **argv)
{
    printf("=============================================================================\n");
    printf("TMA (Tensor Memory Accelerator) Grid-Stride Copy Test - 32KB per chunk\n");
    printf("=============================================================================\n");

    
    cudaDeviceProp prop;
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9)
    {
        printf("❌ TMA requires compute capability >= 9.0 (Hopper architecture)\n");
        printf("Current GPU does not support TMA.\n");
        return 1;
    }
    printf("✅ TMA support detected\n");

    // Parameters - 默认更大的数据量以测试多轮copy
    int N = argc > 1 ? atoi(argv[1]) : 256 * 1024;  // 默认256K个float = 1MB

    // TMA requires 16-byte alignment for some operations, good practice to align.
    N = (N + 3) & ~3; 
    printf("Total elements: N = %d floats\n", N);

    size_t size = N * sizeof(float);
    printf("Total data size: %.1f KB (%.1f MB)\n", size / 1024.0f, size / 1024.0f / 1024.0f);
    
    // 配置grid：每个block处理32KB chunks
    const int elements_per_chunk = 8192;  // 32KB / 4 bytes = 8192 floats
    const int chunk_size_kb = 32;
    
    // 计算需要的总chunk数量
    int total_chunks = (N + elements_per_chunk - 1) / elements_per_chunk;
    
    // 设置block数量：可以小于总chunk数，利用grid-stride loop
    // 建议设置为SM数量的倍数以获得好的负载均衡
    int num_sms = prop.multiProcessorCount;
    int num_blocks = min(total_chunks, num_sms * 4);  // 每个SM最多4个block
    
    printf("\nConfiguration:\n");
    printf("  Elements per chunk: %d (%dKB)\n", elements_per_chunk, chunk_size_kb);
    printf("  Total chunks needed: %d\n", total_chunks);
    printf("  Number of SMs: %d\n", num_sms);
    printf("  Number of blocks: %d\n", num_blocks);
    printf("  Threads per block: 1024\n");
    printf("  Each block will process ~%.1f chunks on average\n", (float)total_chunks / num_blocks);

    // Allocate memory
    float *h_src, *h_dst;
    void *d_src, *d_dst;

    h_src = (float *)malloc(size);
    h_dst = (float *)malloc(size);
    CUDA_CHECK(cudaMalloc(&d_src, size));
    CUDA_CHECK(cudaMalloc(&d_dst, size));

    // Initialize data
    for (int i = 0; i < N; i++)
    {
        h_src[i] = (float)i;
    }
    memset(h_dst, 0, size);

    CUDA_CHECK(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, size));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_runs = 50;
    float times[num_runs];

    // Set shared memory size - 每个block需要32KB用于chunk缓存
    size_t smem_size = elements_per_chunk * sizeof(float);  // 32KB
    printf("\nShared memory per block: %zu bytes (%dKB)\n", smem_size, chunk_size_kb);
    
    if (smem_size > (size_t)prop.sharedMemPerBlock) {
        printf("Error: Requested shared memory size %zu bytes is larger than max %d bytes\n", 
               smem_size, prop.sharedMemPerBlock);
        return 1;
    }
    
    // 设置kernel属性
    CUDA_CHECK(cudaFuncSetAttribute(tma_bulk_copy_kernel, 
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                   smem_size));
    
    printf("\nRunning %d TMA grid-stride tests...\n", num_runs);

    // Warm-up run
    printf("Warming up...\n");
    tma_bulk_copy_kernel<<<num_blocks, 1024, smem_size>>>(d_dst, d_src, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Test
    for (int i = 0; i < num_runs; i++)
    {
        CUDA_CHECK(cudaEventRecord(start));
        tma_bulk_copy_kernel<<<num_blocks, 1024, smem_size>>>(d_dst, d_src, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        if (i < 10 || i % 10 == 9)
        {
            printf("Run %3d: %.3f ms\n", i + 1, times[i]);
        }
    }

    // Calculate statistics
    float min_time = times[0];
    float max_time = times[0];
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++)
    {
        total_time += times[i];
        if (times[i] < min_time)
            min_time = times[i];
        if (times[i] > max_time)
            max_time = times[i];
    }

    float avg_time = total_time / num_runs;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        float diff = times[i] - avg_time;
        variance += diff * diff;
    }
    float std_dev = sqrt(variance / num_runs);

    // Verify result
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost));

    bool correct = true;
    int check_count = min(1000, N);

    for (int i = 0; i < check_count; i++)
    {
        if (h_dst[i] != h_src[i])
        {
            correct = false;
            printf("❌ Mismatch at %d: expected %.0f, got %.0f\n", i, h_src[i], h_dst[i]);
            break;
        }
    }

    if (correct && N > check_count)
    {
        for (int i = N - check_count; i < N; i++)
        {
            if (h_dst[i] != h_src[i])
            {
                correct = false;
                printf("❌ Mismatch at %d: expected %.0f, got %.0f\n", i, h_src[i], h_dst[i]);
                break;
            }
        }
    }

    if (correct)
    {
        printf("✅ TMA data verification passed! (checked %d elements)\n", min(2 * check_count, N));
    }

    printf("\n=== Grid-Stride TMA Results ===\n");
    printf("Data size: %.1f KB (%d floats)\n", size / 1024.0f, N);
    printf("Chunk size: %dKB (%d floats)\n", chunk_size_kb, elements_per_chunk);
    printf("Total chunks: %d, Blocks: %d\n", total_chunks, num_blocks);
    printf("Time (ms):\n");
    printf("  Min:    %.3f\n", min_time);
    printf("  Max:    %.3f\n", max_time);
    printf("  Avg:    %.3f ± %.3f\n", avg_time, std_dev);

    double bandwidth = (size * 2 / (avg_time / 1000.0)) / 1e9; // read+write
    printf("Bandwidth: %.1f GB/s\n", bandwidth);
    printf("Coefficient of Variation: %.2f%%\n", (std_dev / avg_time) * 100.0f);
    printf("✅ Using Hardware TMA (Tensor Memory Accelerator) with grid-stride loop\n");

    // Cleanup
    free(h_src);
    free(h_dst);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n=== Grid-Stride TMA Test Complete ===\n");
    return 0;
}
