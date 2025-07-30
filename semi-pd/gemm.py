import torch

def main():
    # 矩阵尺寸：4096x4096x4096
    
    m = 4096
    k = 8096
    n = 1024 * 32
    # 使用BF16
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"矩阵乘法: A({m}x{k}) × B({k}x{n}) = C({m}x{n})")
    print(f"数据类型: {dtype}")
    
    # 生成随机矩阵
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)
    
    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    C = torch.mm(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)  # ms
    
    # 计算性能
    flops = 2 * m * n * k
    gflops = flops / (elapsed_time * 1e6)  # elapsed_time是ms，转换为GFLOPS
    
    print(f"时间: {elapsed_time:.3f} ms")
    print(f"性能: {gflops:.2f} GFLOPS")
    print("完成!")

if __name__ == "__main__":
    main() 
