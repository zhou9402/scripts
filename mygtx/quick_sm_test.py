#!/usr/bin/env python3
"""
GQA Prefill SM 缩放测试
测试 GQA (Grouped Query Attention) prefill 在不同SM数量下的性能
配置: Q:KV = 1:8 (8个query heads共享1个KV head), KV长度 = 4096
使用 green_context_simple 库 (从 gtx.cpp 编译)
"""

import torch
import flashinfer
import green_context_simple as gc

def setup_data(qo_len=1):
    """
    设置GQA prefill测试数据
    - GQA比例: 64 query heads : 8 KV heads = 8:1 (每8个query heads共享1个KV head)
    - KV长度: 4096 (固定)
    - Query长度: 可变 (测试不同prefill长度)
    """
    kv_len = 4096  # 固定KV长度
    num_kv_heads = 8    # KV heads
    num_qo_heads = 64   # Query heads (8:1 ratio for GQA)
    head_dim = 128
    
    # KV缓存数据
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    
    # Query数据 - prefill格式
    q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0)
    
    print(f"=== GQA Prefill 数据配置 ===")
    print(f"  Query length: {qo_len}")
    print(f"  KV length: {kv_len}")
    print(f"  Query heads: {num_qo_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  GQA ratio (Q:KV): {num_qo_heads//num_kv_heads}:1")
    print(f"  Head dim: {head_dim}")
    print(f"  Total query tokens: {qo_len}")
    
    return q, k, v

def benchmark_single_prefill(q, k, v, sm_count=None, runs=10):
    """单次prefill基准测试"""
    manager = None
    primary_stream_handle = None
    
    try:
        if sm_count:
            # 创建Green Context Manager
            manager = gc.GreenContextManager(device_id=0)
            
            # 创建Green Context和streams
            primary_stream, remaining_stream = manager.create_green_context_and_streams(
                intended_primary_partition_sm_count=sm_count,
                primary_stream_priority=-1,  # 高优先级
                remaining_stream_priority=0   # 正常优先级
            )
            
            # 获取实际的SM分配
            primary_sms, remaining_sms = manager.get_sm_counts()
            print(f"  实际SM分配: Primary={primary_sms}, Remaining={remaining_sms}")
            
            # 将stream handle转换为PyTorch stream对象
            primary_stream_handle = torch.cuda.Stream(stream_ptr=primary_stream)
            print(f"  Primary stream: 0x{primary_stream:x}")
        
            with torch.cuda.stream(primary_stream_handle):
                # 预热
                for _ in range(5):
                    _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
                torch.cuda.synchronize()
                
                # 测试
                times = []
                for _ in range(runs):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record(primary_stream_handle)
                    _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
                    end.record(primary_stream_handle)
                    
                    # 同步primary stream
                    primary_stream_handle.synchronize()
                    
                    times.append(start.elapsed_time(end))
        else:
            # 无Green Context限制，使用默认stream
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
    
    except Exception as e:
        print(f"  错误: {e}")
        return None
    
    finally:
        if manager:
            # 清理资源
            manager.destroy_streams()

def test_standalone_api(sm_count):
    """测试独立API函数"""
    try:
        # 使用独立函数创建Green Context
        primary_stream, remaining_stream = gc.create_green_context_and_streams(
            intended_primary_partition_sm_count=sm_count,
            primary_stream_priority=-1,
            remaining_stream_priority=0,
            device_id=0
        )
        
        print(f"  独立API成功: Primary=0x{primary_stream:x}, Remaining=0x{remaining_stream:x}")
        return True
        
    except Exception as e:
        print(f"  独立API失败: {e}")
        return False

def main():
    print("=" * 70)
    print("GQA Prefill SM 缩放测试")
    print("=" * 70)
    
    # 检查库版本
    try:
        print(f"\n库信息:")
        print(f"  Green Context库版本: {gc.__version__}")
        print(f"  Green Context支持状态: {'支持' if hasattr(gc, 'is_green_context_supported') and gc.is_green_context_supported() else '不支持(使用fallback)'}")
    except:
        print("\n无法获取库信息")
    
    # 固定配置
    qo_len = 4096  # Query长度 (prefill阶段)
    
    print("\n" + "=" * 70)
    q, k, v = setup_data(qo_len=qo_len)
    print("=" * 70)
    
    # 测试的SM数量范围
    sm_counts = [8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 176, 188]
    
    print(f"\n测试SM数量范围: {sm_counts}")
    print(f"每个配置运行10次取平均值")
    
    print("\n" + "-" * 70)
    print("步骤 1/2: 测试基线性能 (无SM限制)")
    print("-" * 70)
    baseline = benchmark_single_prefill(q, k, v, sm_count=None)
    if baseline is None:
        print("❌ 基线测试失败")
        return
        
    print(f"✓ 基线时间: {baseline:.3f} ms")
    baseline_tokens_per_ms = qo_len / baseline
    print(f"✓ 基线吞吐: {baseline_tokens_per_ms:.1f} tokens/ms")
    
    print("\n" + "-" * 70)
    print("步骤 2/2: 测试不同SM数量下的性能")
    print("-" * 70)
    print(f"{'SM数量':<8} {'时间(ms)':<12} {'相对时间':<12} {'相对吞吐':<12} {'吞吐(tok/ms)':<15}")
    print("-" * 70)
    
    for sm_count in sm_counts:
        print(f"测试 {sm_count:3d} SMs...", end="", flush=True)
        time_ms = benchmark_single_prefill(q, k, v, sm_count=sm_count)
        if time_ms:
            ratio = time_ms / baseline
            throughput_ratio = baseline / time_ms
            tokens_per_ms = qo_len / time_ms
            print(f"\r{sm_count:<8} {time_ms:<12.3f} {ratio:<12.2f}x {throughput_ratio:<12.2f}x {tokens_per_ms:<15.1f}")
        else:
            print(f"\r{sm_count:<8} {'Failed':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  GQA配置: 64 query heads, 8 KV heads (8:1)")
    print(f"  Query length: {qo_len} tokens")
    print(f"  KV length: 4096 tokens")
    print(f"  基线性能: {baseline:.3f} ms ({baseline_tokens_per_ms:.1f} tokens/ms)")
    print(f"  使用库: green_context_simple (gtx.cpp)")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"错误: 无法导入 green_context_simple 库")
        print(f"详细错误: {e}")
        print("请确保:")
        print("1. .so 文件在当前目录或Python路径中")
        print("2. 库已正确编译")
        print("3. Python版本匹配")
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc() 