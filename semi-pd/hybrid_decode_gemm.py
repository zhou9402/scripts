"""
混合Decode + GEMM测试
结合FlashInfer decode操作和PyTorch GEMM (M=128, N=8192, K=16384)
"""

import torch
import flashinfer
import green_context_lib as green_context
import argparse
import torch.multiprocessing as mp
import time

# FlashInfer 配置参数
page_size = 16
num_kv_heads = 8   # GQA: KV heads
num_qo_heads = 64  # GQA: Query heads
head_dim = 128

# GEMM 配置参数
GEMM_M = 128
GEMM_N = 8192
GEMM_K = 16384

def setup_decode_data(batch_size=64, kv_len=4096):
    """设置 Decode 测试数据 (qo_len=1)"""
    # Query: NHD layout for decode (single token)
    total_q_len = batch_size * 1  # decode时qo_len=1
    q = torch.randn(total_q_len, num_qo_heads, head_dim).half().to("cuda:0")

    # 计算所需的页面数
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    
    # Paged KV Cache: NHD layout [num_pages, 2, page_size, num_kv_heads, head_dim]
    paged_kv_cache = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim).half().to("cuda:0")
    
    # KV indices: 每个请求使用连续的页面
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda:0")
    
    # KV indptr: 指向每个序列的页面范围
    paged_kv_indptr = torch.arange(0, total_pages + 1, pages_per_seq, dtype=torch.int32, device="cuda:0")
    
    # KV last page length: 每个序列最后一页的有效长度
    last_page_len = kv_len % page_size
    if last_page_len == 0:
        last_page_len = page_size
    paged_kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device="cuda:0")
    
    # QO indptr: 每个序列的查询范围 (decode时每个序列只有1个token)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda:0")

    return q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr

def setup_gemm_data():
    """设置 GEMM 测试数据 (M=128, N=8192, K=16384)"""
    A = torch.randn(GEMM_M, GEMM_K, dtype=torch.float32, device="cuda:0")
    B = torch.randn(GEMM_K, GEMM_N, dtype=torch.float32, device="cuda:0")
    return A, B

def decode_worker(q, kv_cache, runs, sm_count, ready_event, result_queue,
                  qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len):
    """在green context下运行decode的进程函数"""
    # 在进程内部分配资源并plan
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    wrapper.plan(qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr, paged_kv_indices=paged_kv_indices,
                         paged_kv_last_page_len=paged_kv_last_page_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                         head_dim_qk=head_dim, page_size=page_size, causal=False)  # decode用False

    print("Decode worker: 等待开始...")
    if not green_context.create_green_context(sm_count=sm_count):
        print(f"Decode worker: 无法创建Green Context ({sm_count} SMs)")
        result_queue.put(("decode", -1))
        return
    if not green_context.switch_to_green_context():
        print(f"Decode worker: 无法切换到Green Context")
        green_context.destroy_green_context()
        result_queue.put(("decode", -1))
        return
    
    try:
        # 预热
        for _ in range(2):
            wrapper.run(q, kv_cache)
        torch.cuda.synchronize()

        ready_event.wait() # 等待主线程信号
        print(f"Decode worker: 开始执行 (在 {sm_count} SMs Green Context中)")
        
        # 计时开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        total_runs = runs * 100  # decode运行更多次以确保重叠
        start_event.record()
        for _ in range(total_runs):
            wrapper.run(q, kv_cache)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        result_queue.put(("decode", elapsed_time))
        print(f"Decode worker: 完成 {total_runs} 次运行，耗时: {elapsed_time:.3f} ms")
    finally:
        green_context.destroy_green_context()
        print("Decode worker: 已销毁Green Context")

def gemm_worker(A, B, runs, sm_count, ready_event, result_queue):
    """在green context下运行GEMM的进程函数"""
    print("GEMM worker: 等待开始...")
    if not green_context.create_green_context(sm_count=sm_count):
        print(f"GEMM worker: 无法创建Green Context ({sm_count} SMs)")
        result_queue.put(("gemm", -1))
        return
    if not green_context.switch_to_green_context():
        print(f"GEMM worker: 无法切换到Green Context")
        green_context.destroy_green_context()
        result_queue.put(("gemm", -1))
        return
    
    try:
        # 预热
        for _ in range(2):
            C = torch.mm(A, B)
        torch.cuda.synchronize()

        ready_event.wait() # 等待主线程信号
        print(f"GEMM worker: 开始执行 (在 {sm_count} SMs Green Context中)")
        
        # 计时开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            C = torch.mm(A, B)  # PyTorch GEMM: C = A @ B
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        result_queue.put(("gemm", elapsed_time))
        print(f"GEMM worker: 完成 {runs} 次运行，耗时: {elapsed_time:.3f} ms")
    finally:
        green_context.destroy_green_context()
        print("GEMM worker: 已销毁Green Context")

def test_concurrent_hybrid(q, kv_cache, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
                          A, B, runs, sm_decode=40, sm_gemm=30):
    """并发执行Decode和GEMM测试"""
    print("\n=== 并发混合测试 ===")
    print(f"Decode SMs: {sm_decode}, GEMM SMs: {sm_gemm}")
    
    # 为CUDA + multiprocessing 设置 'spawn' 启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 创建进程通信
    ready_event = mp.Event()
    result_queue = mp.Queue()

    # 创建decode进程
    decode_args = (
        q, kv_cache, runs, sm_decode, ready_event, result_queue,
        qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len
    )
    decode_process = mp.Process(target=decode_worker, args=decode_args)

    # 创建gemm进程
    gemm_args = (
        A, B, runs, sm_gemm, ready_event, result_queue
    )
    gemm_process = mp.Process(target=gemm_worker, args=gemm_args)

    # 总体计时
    overall_start_event = torch.cuda.Event(enable_timing=True)
    overall_end_event = torch.cuda.Event(enable_timing=True)
    
    overall_start_event.record()
    
    # 启动进程
    decode_process.start()
    gemm_process.start()
    
    # 等待worker初始化完成
    time.sleep(3) 
    
    ready_event.set() # 发送开始信号
    
    # 等待完成
    decode_process.join()
    gemm_process.join()
    
    overall_end_event.record()
    torch.cuda.synchronize()
    
    # 收集结果
    results = {}
    for _ in range(2):  # 收集两个进程的结果
        worker_type, elapsed_time = result_queue.get()
        results[worker_type] = elapsed_time
    
    overall_time = overall_start_event.elapsed_time(overall_end_event)
    results['total'] = overall_time
    
    print(f"并发执行总时间: {overall_time:.3f} ms")
    print(f"Decode时间: {results.get('decode', -1):.3f} ms")
    print(f"GEMM时间: {results.get('gemm', -1):.3f} ms")
    
    return results

def test_sequential_hybrid(q, kv_cache, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
                          A, B, runs):
    """顺序执行Decode和GEMM测试"""
    print("\n=== 顺序混合测试 ===")
    
    # 设置decode wrapper
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    wrapper.plan(qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr, paged_kv_indices=paged_kv_indices,
                 paged_kv_last_page_len=paged_kv_last_page_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                 head_dim_qk=head_dim, page_size=page_size, causal=False)
    
    # 预热
    for _ in range(2):
        wrapper.run(q, kv_cache)
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # 测试Decode (运行更多次匹配并发模式)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_decode_runs = runs * 100
    start_event.record()
    for _ in range(total_decode_runs):
        wrapper.run(q, kv_cache)
    end_event.record()
    torch.cuda.synchronize()
    
    decode_time = start_event.elapsed_time(end_event)
    
    # 测试GEMM
    start_event.record()
    for _ in range(runs):
        C = torch.mm(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    gemm_time = start_event.elapsed_time(end_event)
    
    total_time = decode_time + gemm_time
    
    print(f"Decode时间: {decode_time:.3f} ms")
    print(f"GEMM时间: {gemm_time:.3f} ms")
    print(f"总时间: {total_time:.3f} ms")
    
    return {"decode": decode_time, "gemm": gemm_time, "total": total_time}

def main():
    parser = argparse.ArgumentParser(description='混合Decode + GEMM测试')
    parser.add_argument('--mode', choices=['simple', 'hybrid', 'both'], default='simple', 
                       help='测试模式: simple=仅GEMM, hybrid=混合测试, both=两者都测试')
    parser.add_argument('--sm_decode', type=int, default=40, help='Decode使用的SM数量')
    parser.add_argument('--sm_gemm', type=int, default=30, help='GEMM使用的SM数量')
    parser.add_argument('--batch_size', type=int, default=32, help='Decode的batch size')
    parser.add_argument('--runs', type=int, default=10, help='运行次数')
    args = parser.parse_args()

    print(f"=== 混合Decode + GEMM测试 ===")
    print(f"测试模式: {args.mode}")
    print(f"GEMM维度: M={GEMM_M}, N={GEMM_N}, K={GEMM_K}")
    
    if args.mode in ['simple', 'both']:
        # 简化测试 - 仅测试GEMM性能
        print("\n=== GEMM性能测试 ===")
        
        A, B = setup_gemm_data()
        data_size = (GEMM_M * GEMM_K + GEMM_K * GEMM_N + GEMM_M * GEMM_N) * 4  # bytes
        print(f"GEMM数据量: {data_size/1024/1024:.1f} MB")
        
        # 预热
        for _ in range(2):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # 性能测试
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(args.runs):
            C = torch.mm(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed = start_event.elapsed_time(end_event)
        
        # 计算FLOPS
        flops_per_gemm = 2 * GEMM_M * GEMM_N * GEMM_K  # 每次GEMM的浮点运算数
        total_flops = flops_per_gemm * args.runs
        gflops = (total_flops / 1e9) / (elapsed / 1000)
        
        print(f"GEMM时间: {elapsed:.3f} ms")
        print(f"GEMM性能: {gflops:.1f} GFLOPS")
    
    if args.mode in ['hybrid', 'both']:
        # 混合测试 - Decode + GEMM
        print(f"\n=== 混合测试 ===")
        
        # 设置数据
        q, kv_cache, kv_indices, kv_indptr, kv_last_len, qo_indptr = setup_decode_data(
            batch_size=args.batch_size, kv_len=4096)
        A, B = setup_gemm_data()
        
        # 数据量对比
        decode_data = q.numel() * 2 + kv_cache.numel() * 2  # 估算读写量 (bytes)
        gemm_data = (GEMM_M * GEMM_K + GEMM_K * GEMM_N + GEMM_M * GEMM_N) * 4  # bytes
        
        print(f"Decode数据量: ~{decode_data/1024/1024:.1f} MB (内存密集)")
        print(f"GEMM数据量: ~{gemm_data/1024/1024:.1f} MB (计算密集)")
        
        # 顺序测试
        sequential_results = test_sequential_hybrid(
            q, kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_len,
            A, B, args.runs)
        
        # 并发测试
        concurrent_results = test_concurrent_hybrid(
            q, kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_len,
            A, B, args.runs, args.sm_decode, args.sm_gemm)
        
        # 性能对比
        print("\n=== 性能对比分析 ===")
        if sequential_results and concurrent_results:
            seq_total = sequential_results['total']
            conc_total = concurrent_results['total']
            
            print(f"顺序执行总时间: {seq_total:.3f} ms")
            print(f"并发执行总时间: {conc_total:.3f} ms")
            print(f"加速比: {seq_total/conc_total:.2f}x")
            
            overlap_efficiency = (seq_total - conc_total) / seq_total * 100
            print(f"重叠效率: {overlap_efficiency:.1f}%")
            
            # 工作负载分析
            decode_time = concurrent_results.get('decode', 0)
            gemm_time = concurrent_results.get('gemm', 0)
            
            if decode_time > 0 and gemm_time > 0:
                print(f"\n--- 工作负载特性 ---")
                print(f"Decode (内存密集): {decode_time:.3f} ms")
                print(f"GEMM (计算密集): {gemm_time:.3f} ms")
                balance = min(decode_time, gemm_time)/max(decode_time, gemm_time)
                print(f"工作负载平衡度: {balance:.2f}")
                
                if balance > 0.8:
                    print("✅ 工作负载平衡良好，重叠效果最佳")
                elif balance > 0.5:
                    print("⚠️  工作负载不够平衡，可调整SM分配")
                else:
                    print("❌ 工作负载严重不平衡，建议调整参数")

if __name__ == "__main__":
    main()
