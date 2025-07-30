"""
测试高计算密度、低数据传输量的kernel
这个kernel专门设计用于测试计算密集型工作负载
"""

import torch
import green_context_lib as green_context_lib
import time

def test_compute_intensive_kernel():
    """测试计算密集型kernel"""
    print("=== 计算密集型Kernel测试 ===")
    
    # 编译库
    print("编译green_context库...")
    green_context_lib.create_green_context()
    
    # 检查GPU信息
    device_prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_prop.name}")
    print(f"Compute Capability: {device_prop.major}.{device_prop.minor}")
    print(f"Streaming Multiprocessors: {device_prop.multi_processor_count}")
    
    # 准备输入参数（数据量很小，只有4个float）
    input_params = torch.tensor([
        1.0,    # base_val
        2.0,    # multiplier  
        0.5,    # offset
        3.14    # frequency
    ], dtype=torch.float32, device='cuda')
    
    print(f"输入数据大小: {input_params.numel() * 4} bytes (只有 {input_params.numel()} 个float)")
    
    # 测试不同的计算强度
    test_iterations = [100, 500, 1000, 2000, 5000]
    
    print("\n--- 性能测试：不同计算强度 ---")
    print("计算强度 | 执行时间 | GFLOPS")
    print("-" * 35)
    
    for iterations in test_iterations:
        # 预热
        green_context_lib.run_compute_intensive_kernel(input_params, 10)
        torch.cuda.synchronize()
        
        # 计时测试
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_runs = 5
        total_time = 0
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_event.record()
            green_context_lib.run_compute_intensive_kernel(input_params, iterations)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed = start_event.elapsed_time(end_event)
            total_time += elapsed
        
        avg_time = total_time / num_runs
        
        # 估算FLOPS：每个线程每次迭代约15个浮点运算
        # 总线程数：64 blocks * 256 threads = 16384 threads
        total_threads = 64 * 256
        ops_per_iteration = 15  # 估算的浮点运算数
        total_ops = total_threads * iterations * ops_per_iteration
        gflops = (total_ops / 1e9) / (avg_time / 1000)
        
        print(f"{iterations:8d} | {avg_time:8.3f} ms | {gflops:6.1f}")

def test_green_context_integration():
    """测试与Green Context的集成"""
    print("\n=== Green Context集成测试 ===")
    
    input_params = torch.tensor([1.0, 2.0, 0.5, 3.14], dtype=torch.float32, device='cuda')
    iterations = 1000
    
    # 测试不同的SM配置
    sm_configs = [8, 16, 32, 64]
    
    print("SM数量 | 执行时间 | 性能")
    print("-" * 25)
    
    for sm_count in sm_configs:
        try:
            # 创建Green Context
            if green_context_lib.create_green_context(sm_count):
                if green_context_lib.switch_to_green_context():
                    # 预热
                    green_context_lib.run_compute_intensive_kernel(input_params, 10)
                    torch.cuda.synchronize()
                    
                    # 性能测试
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    torch.cuda.synchronize()
                    start_event.record()
                    green_context_lib.run_compute_intensive_kernel(input_params, iterations)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    elapsed = start_event.elapsed_time(end_event)
                    
                    # 计算相对性能（以8 SM为基准）
                    if sm_count == 8:
                        baseline_time = elapsed
                        relative_perf = 1.0
                    else:
                        relative_perf = baseline_time / elapsed
                    
                    print(f"{sm_count:6d} | {elapsed:8.3f} ms | {relative_perf:4.2f}x")
                    
                    green_context_lib.switch_to_default_context()
                
                green_context_lib.destroy_green_context()
                
        except Exception as e:
            print(f"{sm_count:6d} | 失败: {e}")

def test_compute_vs_memory_bandwidth():
    """比较计算密集型kernel和内存带宽限制的操作"""
    print("\n=== 计算密集 vs 内存带宽测试 ===")
    
    # 1. 计算密集型kernel
    input_params = torch.tensor([1.0, 2.0, 0.5, 3.14], dtype=torch.float32, device='cuda')
    iterations = 2000
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 测试计算密集型kernel
    torch.cuda.synchronize()
    start_event.record()
    green_context_lib.run_compute_intensive_kernel(input_params, iterations)
    end_event.record()
    torch.cuda.synchronize()
    
    compute_time = start_event.elapsed_time(end_event)
    
    # 2. 内存带宽测试（简单的tensor复制）
    size = 32 * 1024 * 1024  # 128MB数据
    src_tensor = torch.randn(size, dtype=torch.float32, device='cuda')
    
    torch.cuda.synchronize()
    start_event.record()
    dst_tensor = src_tensor.clone()
    end_event.record()
    torch.cuda.synchronize()
    
    memory_time = start_event.elapsed_time(end_event)
    memory_bandwidth = (size * 4 * 2 / 1e9) / (memory_time / 1000)  # GB/s
    
    print(f"计算密集型kernel: {compute_time:.3f} ms")
    print(f"内存带宽测试: {memory_time:.3f} ms ({memory_bandwidth:.1f} GB/s)")
    print(f"计算/内存时间比: {compute_time/memory_time:.2f}")
    
    # 数据传输量对比
    compute_data = input_params.numel() * 4  # bytes
    memory_data = size * 4 * 2  # bytes (read + write)
    
    print(f"计算kernel数据量: {compute_data} bytes")
    print(f"内存测试数据量: {memory_data/1024/1024:.1f} MB")
    print(f"数据量比例: 1:{memory_data//compute_data}")

def main():
    print("=== 高计算密度Kernel完整测试 ===\n")
    
    try:
        test_compute_intensive_kernel()
        test_green_context_integration()
        test_compute_vs_memory_bandwidth()
        
        print("\n=== 测试总结 ===")
        print("✅ 计算密集型kernel测试完成")
        print("✅ Green Context集成测试完成") 
        print("✅ 计算vs内存带宽对比完成")
        print("\n这个kernel设计特点：")
        print("- 数据加载量极小（仅4个float，16字节）")
        print("- 计算密度极高（大量数学运算）")
        print("- 适合测试纯计算性能")
        print("- 支持Green Context资源隔离")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 