#!/usr/bin/env python3
"""
快速SM缩放测试 - 单次Prefill版本
测试单次prefill在不同SM数量下的性能
"""

import torch
import flashinfer
import green_context_lib as green_context

def setup_data(qo_len=1):
    """设置单次prefill测试数据"""
    kv_len = 4096
    num_kv_heads = 8
    num_qo_heads = 64
    head_dim = 128
    
    # KV缓存数据
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    
    # Query数据 - 单次prefill格式
    q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0)
    
    print(f"数据配置:")
    print(f"  Query length: {qo_len}")
    print(f"  KV length: {kv_len}")
    print(f"  Query heads: {num_qo_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total tokens: {qo_len}")
    
    return q, k, v

def benchmark_single_prefill(q, k, v, sm_count=None, runs=10):
    """单次prefill基准测试"""
    if sm_count:
        success = green_context.create_green_context(sm_count=sm_count)
        if not success:
            print(f"Failed to create Green Context with {sm_count} SMs")
            return None
    
    try:
        # 预热
        for _ in range(5):
            _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        torch.cuda.synchronize()
        
        # 测试
        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
        
        avg_time = sum(times) / len(times)
        return avg_time
    
    finally:
        if sm_count:
            green_context.destroy_green_context()

def main():
    print("=== 单次 Prefill SM 缩放测试 ===\n")
    
    # 固定配置
    qo_len = 128
    
    print("设置测试数据...")
    q, k, v = setup_data(qo_len=qo_len)
    
    # 测试的SM数量
    sm_counts = [8, 16, 32, 64, 72, 80, 96, 112, 128, 144, 160, 176, 188]
    
    print("\n测试正常模式...")
    baseline = benchmark_single_prefill(q, k, v, sm_count=None)
    if baseline is None:
        print("正常模式测试失败")
        return
        
    print(f"正常模式 (无限制): {baseline:.3f} ms")
    
    print(f"\n测试不同SM数量:")
    print(f"{'SM数量':<8} {'时间(ms)':<12} {'vs基线':<10} {'吞吐提升':<10}")
    print("-" * 50)
    
    for sm_count in sm_counts:
        print(f"测试 {sm_count} SMs...")
        time_ms = benchmark_single_prefill(q, k, v, sm_count=sm_count)
        if time_ms:
            ratio = time_ms / baseline
            throughput_boost = baseline / time_ms
            print(f"{sm_count:<8} {time_ms:<12.3f} {ratio:<10.2f}x {throughput_boost:<10.2f}x")
        else:
            print(f"{sm_count:<8} {'Failed':<12} {'N/A':<10} {'N/A':<10}")
    
    print(f"\n性能总结:")
    print(f"  Query tokens: {qo_len}")
    print(f"  Baseline tokens/ms: {qo_len/baseline:.1f}")

if __name__ == "__main__":
    main() 