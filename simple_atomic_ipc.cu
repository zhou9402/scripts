#include <iostream>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <signal.h>

// Helper to check for CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Named pipes for IPC
const char* PIPE_COUNTER_P2 = "/tmp/atomic_counter_handle_p2";
const char* PIPE_COUNTER_P3 = "/tmp/atomic_counter_handle_p3";
const char* PIPE_SYNC = "/tmp/atomic_sync";

// System-level atomic add kernel for cross-process operations
__global__ void atomicAddKernel(unsigned int* counter, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = 0; i < iterations; i++) {
        // Use system-level atomic operation for cross-process atomicity
        asm volatile("atom.global.sys.add.u32 _, [%0], %1;" 
                     : 
                     : "l"(counter), "r"(1u) 
                     : "memory");
    }
}

void cleanup_pipes() {
    unlink(PIPE_COUNTER_P2);
    unlink(PIPE_COUNTER_P3);
    unlink(PIPE_SYNC);
}

void signal_handler(int sig) {
    cleanup_pipes();
    exit(sig);
}

// Process 1: Creates counter and waits for others
void run_process1(int iterations) {
    std::cout << "=== Process 1 (Counter Owner) ===" << std::endl;
    
    CHECK_CUDA(cudaSetDevice(0));
    
    // Allocate and initialize counter
    unsigned int* counter;
    CHECK_CUDA(cudaMalloc(&counter, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(counter, 0, sizeof(unsigned int)));
    
    // Get IPC handle
    cudaIpcMemHandle_t counter_handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&counter_handle, counter));
    
    std::cout << "创建共享计数器，初始值: 0" << std::endl;
    
    // Create named pipes
    cleanup_pipes();
    if (mkfifo(PIPE_COUNTER_P2, 0666) != 0) {
        perror("mkfifo counter_p2");
        exit(1);
    }
    if (mkfifo(PIPE_COUNTER_P3, 0666) != 0) {
        perror("mkfifo counter_p3");
        exit(1);
    }
    if (mkfifo(PIPE_SYNC, 0666) != 0) {
        perror("mkfifo sync");
        exit(1);
    }
    
    std::cout << "等待其他进程连接..." << std::endl;
    
    // Create ready pipe first
    const char* PIPE_READY = "/tmp/atomic_ready";
    unlink(PIPE_READY);  // Remove if exists
    if (mkfifo(PIPE_READY, 0666) != 0) {
        perror("mkfifo ready");
        exit(1);
    }
    
    // Send counter handle to each process using separate pipes
    std::cout << "等待Process 2连接..." << std::endl;
    int fd_counter_p2 = open(PIPE_COUNTER_P2, O_WRONLY);
    if (fd_counter_p2 == -1) {
        perror("open counter pipe p2");
        exit(1);
    }
    if (write(fd_counter_p2, &counter_handle, sizeof(counter_handle)) != sizeof(counter_handle)) {
        perror("write counter handle p2");
        exit(1);
    }
    close(fd_counter_p2);
    std::cout << "计数器句柄已发送给Process 2" << std::endl;
    
    std::cout << "等待Process 3连接..." << std::endl;
    int fd_counter_p3 = open(PIPE_COUNTER_P3, O_WRONLY);
    if (fd_counter_p3 == -1) {
        perror("open counter pipe p3");
        exit(1);
    }
    if (write(fd_counter_p3, &counter_handle, sizeof(counter_handle)) != sizeof(counter_handle)) {
        perror("write counter handle p3");
        exit(1);
    }
    close(fd_counter_p3);
    std::cout << "计数器句柄已发送给Process 3" << std::endl;
    
    // Wait for all processes to be ready before starting concurrent execution
    std::cout << "等待所有进程准备就绪..." << std::endl;
    
    // Wait for ready signals from other processes
    int fd_ready = open(PIPE_READY, O_RDONLY);
    if (fd_ready == -1) {
        perror("open ready pipe");
        exit(1);
    }
    
    char ready_msg[64];
    int ready_processes = 0;
    while (ready_processes < 2) {
        if (read(fd_ready, ready_msg, sizeof(ready_msg)) > 0) {
            ready_processes++;
            std::cout << "收到就绪信号: " << ready_msg << std::endl;
        }
    }
    close(fd_ready);
    unlink(PIPE_READY);
    
    std::cout << "所有进程就绪，Process 1 作为协调者，不执行原子操作" << std::endl;
    std::cout << "等待Process 2和3完成原子操作..." << std::endl;
    
    // Wait for completion signals from other processes
    std::cout << "等待其他进程完成..." << std::endl;
    
    int fd_sync = open(PIPE_SYNC, O_RDONLY);
    if (fd_sync == -1) {
        perror("open sync pipe");
        exit(1);
    }
    
    char sync_msg[64];
    int completed_processes = 0;
    while (completed_processes < 2) {
        if (read(fd_sync, sync_msg, sizeof(sync_msg)) > 0) {
            completed_processes++;
            std::cout << "收到完成信号: " << sync_msg << std::endl;
        }
    }
    close(fd_sync);
    
    // Read final counter value
    unsigned int final_value;
    CHECK_CUDA(cudaMemcpy(&final_value, counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Calculate expected value (only 2 processes execute atomic operations)
    int blocks = 256;
    int threads = 256;
    int total_ops = blocks * threads * iterations;
    
    std::cout << "\n=== 最终结果 ===" << std::endl;
    std::cout << "最终计数器值: " << final_value << std::endl;
    std::cout << "预期值: " << (2 * total_ops) << " (2个进程 × " << total_ops << " 操作)" << std::endl;
    std::cout << "正确性: " << (final_value == 2 * total_ops ? "✓ 正确" : "✗ 错误") << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaFree(counter));
    cleanup_pipes();
    
    std::cout << "Process 1 完成" << std::endl;
}

// Process 2 and 3: Connect to counter and perform atomic operations
void run_worker_process(int process_id, int gpu_id, int iterations) {
    std::cout << "=== Process " << process_id << " (GPU " << gpu_id << ") ===" << std::endl;
    
    CHECK_CUDA(cudaSetDevice(gpu_id));
    
    std::cout << "等待接收计数器句柄..." << std::endl;
    
    // Choose the correct pipe based on process ID
    const char* pipe_name = (process_id == 2) ? PIPE_COUNTER_P2 : PIPE_COUNTER_P3;
    std::cout << "尝试打开管道: " << pipe_name << std::endl;
    
    // Receive counter handle
    int fd_counter = open(pipe_name, O_RDONLY);
    if (fd_counter == -1) {
        perror("open counter pipe");
        exit(1);
    }
    std::cout << "成功打开计数器管道" << std::endl;
    
    cudaIpcMemHandle_t counter_handle;
    std::cout << "开始读取计数器句柄..." << std::endl;
    if (read(fd_counter, &counter_handle, sizeof(counter_handle)) != sizeof(counter_handle)) {
        perror("read counter handle");
        exit(1);
    }
    close(fd_counter);
    
    std::cout << "接收到计数器句柄" << std::endl;
    
    // Open the counter
    unsigned int* counter;
    CHECK_CUDA(cudaIpcOpenMemHandle((void**)&counter, counter_handle, cudaIpcMemLazyEnablePeerAccess));
    
    // Send ready signal to Process 1
    const char* PIPE_READY = "/tmp/atomic_ready";
    std::cout << "尝试打开就绪管道: " << PIPE_READY << std::endl;
    int fd_ready = open(PIPE_READY, O_WRONLY);
    if (fd_ready == -1) {
        perror("open ready pipe");
        exit(1);
    }
    std::cout << "成功打开就绪管道" << std::endl;
    
    char ready_msg[64];
    snprintf(ready_msg, sizeof(ready_msg), "Process %d ready", process_id);
    if (write(fd_ready, ready_msg, strlen(ready_msg) + 1) == -1) {
        perror("write ready");
    }
    close(fd_ready);
    
    std::cout << "已发送就绪信号，等待并发执行..." << std::endl;
    
    // Small delay to ensure all processes start roughly at the same time
    usleep(100000); // 100ms
    
    std::cout << "开始并发执行原子操作..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int blocks = 256;
    int threads = 256;
    
    CHECK_CUDA(cudaEventRecord(start));
    atomicAddKernel<<<blocks, threads>>>(counter, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    int total_ops = blocks * threads * iterations;
    std::cout << "Process " << process_id << " 完成 " << total_ops 
              << " 次原子操作，耗时: " << elapsed_ms << " ms" << std::endl;
    
    // Send completion signal
    int fd_sync = open(PIPE_SYNC, O_WRONLY);
    if (fd_sync == -1) {
        perror("open sync pipe");
        exit(1);
    }
    
    char completion_msg[64];
    snprintf(completion_msg, sizeof(completion_msg), "Process %d completed", process_id);
    if (write(fd_sync, completion_msg, strlen(completion_msg) + 1) == -1) {
        perror("write sync");
    }
    close(fd_sync);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaIpcCloseMemHandle(counter));
    
    std::cout << "Process " << process_id << " 完成" << std::endl;
}

void print_usage(const char* prog_name) {
    std::cout << "用法:" << std::endl;
    std::cout << "  Process 1: " << prog_name << " 1 [iterations]" << std::endl;
    std::cout << "  Process 2: " << prog_name << " 2 [iterations]" << std::endl;
    std::cout << "  Process 3: " << prog_name << " 3 [iterations]" << std::endl;
    std::cout << std::endl;
    std::cout << "功能:" << std::endl;
    std::cout << "  - 2个进程并发对共享计数器执行系统级原子加法" << std::endl;
    std::cout << "  - 使用 atom.global.sys.add.u32 指令" << std::endl;
    std::cout << "  - Process 1 (GPU 0): 创建计数器，仅作协调者" << std::endl;
    std::cout << "  - Process 2 (GPU 1): 执行原子操作" << std::endl;
    std::cout << "  - Process 3 (GPU 5): 执行原子操作" << std::endl;
    std::cout << std::endl;
    std::cout << "参数:" << std::endl;
    std::cout << "  iterations: 每个线程的迭代次数, 默认1000" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  终端1: " << prog_name << " 1 1000" << std::endl;
    std::cout << "  终端2: " << prog_name << " 2 1000" << std::endl;
    std::cout << "  终端3: " << prog_name << " 3 1000" << std::endl;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    int process_id = std::stoi(argv[1]);
    int iterations = (argc > 2) ? std::stoi(argv[2]) : 1000;
    
    switch (process_id) {
        case 1:
            run_process1(iterations);
            break;
        case 2:
            run_worker_process(2, 1, iterations);
            break;
        case 3:
            run_worker_process(3, 5, iterations);
            break;
        default:
            std::cerr << "无效的进程ID: " << process_id << " (应该是1, 2, 或3)" << std::endl;
            print_usage(argv[0]);
            return 1;
    }
    
    return 0;
} 
