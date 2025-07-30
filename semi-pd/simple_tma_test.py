"""
简单的TMA Load Kernel测试
用于快速验证TMA功能是否正常工作
"""

import torch
import green_context_lib as green_context_lib

def main():
    print("=== 简单TMA测试 ===")
    
    # 编译库
    print("编译green_context库...")
    green_context_lib.create_green_context()
    
    # 检查GPU信息
    device_prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_prop.name}")
    print(f"Compute Capability: {device_prop.major}.{device_prop.minor}")
    
    # 测试数据大小
    test_sizes = [
        (8 * 1024 * 1024, "32MB"),
        (32 * 1024 * 1024, "128MB")
    ]
    
    for size, size_name in test_sizes:
        print(f"\n测试 {size_name}...")
        
        # 创建测试数据
        src_tensor = torch.randn(size, dtype=torch.float32, device='cuda')
        
        # 执行TMA kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        try:
            torch.cuda.synchronize()
            start_event.record()
            green_context_lib.run_tma_load_kernel(src_tensor)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed = start_event.elapsed_time(end_event)
            data_gb = size * 4 / 1e9
            bandwidth = (data_gb * 2) / (elapsed / 1000)
            
            print(f"✅ {size_name}: {elapsed:.3f} ms, {bandwidth:.1f} GB/s")
            
        except Exception as e:
            print(f"❌ {size_name}: 失败 - {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main() 