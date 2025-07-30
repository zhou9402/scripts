"""
混合Prefill + 计算密集型Kernel测试
结合FlashInfer prefill操作和高计算密度的数学kernel
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

def setup_paged_data(batch_size=64, kv_len=4096, qo_len=4096):
    """设置 Paged KV Cache 测试数据 (从hybrid.py复制)"""
    # Query: NHD layout
    total_q_len = batch_size * qo_len
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
    
    # QO indptr: 每个序列的查询范围
    qo_indptr = torch.arange(0, batch_size * qo_len + 1, qo_len, dtype=torch.int32, device="cuda:0")

    return q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr

def prefill_worker(q, kv_cache, runs, sm_count, ready_event, result_queue,
                   qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len):
    """在green context下运行prefill的进程函数"""
    # 在进程内部分配资源并plan
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    wrapper.plan(qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr, paged_kv_indices=paged_kv_indices,
                         paged_kv_last_page_len=paged_kv_last_page_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                         head_dim_qk=head_dim, page_size=page_size, causal=True)

    print("Prefill worker: 等待开始...")
    if not green_context.create_green_context(sm_count=sm_count):
        print(f"Prefill worker: 无法创建Green Context ({sm_count} SMs)")
        result_queue.put(("prefill", -1))
        return
    if not green_context.switch_to_green_context():
        print(f"Prefill worker: 无法切换到Green Context")
        green_context.destroy_green_context()
        result_queue.put(("prefill", -1))
        return
    
    try:
        # 预热
        for _ in range(2):
            wrapper.run(q, kv_cache)
        torch.cuda.synchronize()

        ready_event.wait() # 等待主线程信号
        print(f"Prefill worker: 开始执行 (在 {sm_count} SMs Green Context中)")
        
        # 计时开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            wrapper.run(q, kv_cache)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        result_queue.put(("prefill", elapsed_time))
        print(f"Prefill worker: 完成 {runs} 次运行，耗时: {elapsed_time:.3f} ms")
    finally:
        green_context.destroy_green_context()
        print("Prefill worker: 已销毁Green Context")

def compute_worker(input_params, iterations_per_thread, runs, sm_count, ready_event, result_queue):
    """在green context下运行计算密集型kernel的进程函数"""
    print("Compute worker: 等待开始...")
    if not green_context.create_green_context(sm_count=sm_count):
        print(f"Compute worker: 无法创建Green Context ({sm_count} SMs)")
        result_queue.put(("compute", -1))
        return
    if not green_context.switch_to_green_context():
        print(f"Compute worker: 无法切换到Green Context")
        green_context.destroy_green_context()
        result_queue.put(("compute", -1))
        return
    
    try:
        # 预热
        green_context.run_compute_intensive_kernel(input_params, 10)
        torch.cuda.synchronize()

        ready_event.wait() # 等待主线程信号
        print(f"Compute worker: 开始执行 (在 {sm_count} SMs Green Context中)")
        
        # 计时开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            green_context.run_compute_intensive_kernel(input_params, iterations_per_thread)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        result_queue.put(("compute", elapsed_time))
        print(f"Compute worker: 完成 {runs} 次运行，耗时: {elapsed_time:.3f} ms")
    finally:
        green_context.destroy_green_context()
        print("Compute worker: 已销毁Green Context")

def test_concurrent_hybrid(q, kv_cache, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
                          input_params, iterations_per_thread, runs, sm_prefill=40, sm_compute=30):
    """并发执行Prefill和Compute测试"""
    print("\n=== 并发混合测试 ===")
    print(f"Prefill SMs: {sm_prefill}, Compute SMs: {sm_compute}")
    
    # 为CUDA + multiprocessing 设置 'spawn' 启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 创建进程通信
    ready_event = mp.Event()
    result_queue = mp.Queue()

    # 创建prefill进程
    prefill_args = (
        q, kv_cache, runs, sm_prefill, ready_event, result_queue,
        qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len
    )
    prefill_process = mp.Process(target=prefill_worker, args=prefill_args)

    # 创建compute进程
    compute_args = (
        input_params, iterations_per_thread, runs, sm_compute, ready_event, result_queue
    )
    compute_process = mp.Process(target=compute_worker, args=compute_args)

    # 总体计时
    overall_start_event = torch.cuda.Event(enable_timing=True)
    overall_end_event = torch.cuda.Event(enable_timing=True)
    
    overall_start_event.record()
    
    # 启动进程
    prefill_process.start()
    compute_process.start()
    
    # 等待worker初始化完成
    time.sleep(3) 
    
    ready_event.set() # 发送开始信号
    
    # 等待完成
    prefill_process.join()
    compute_process.join()
    
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
    print(f"Prefill时间: {results.get('prefill', -1):.3f} ms")
    print(f"Compute时间: {results.get('compute', -1):.3f} ms")
    
    return results

def test_sequential_hybrid(q, kv_cache, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
                          input_params, iterations_per_thread, runs):
    """顺序执行Prefill和Compute测试"""
    print("\n=== 顺序混合测试 ===")
    
    # 设置wrapper
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    wrapper.plan(qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr, paged_kv_indices=paged_kv_indices,
                 paged_kv_last_page_len=paged_kv_last_page_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                 head_dim_qk=head_dim, page_size=page_size, causal=True)
    
    # 预热
    for _ in range(2):
        wrapper.run(q, kv_cache)
        green_context.run_compute_intensive_kernel(input_params, 10)
    torch.cuda.synchronize()
    
    # 测试Prefill
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        wrapper.run(q, kv_cache)
    end_event.record()
    torch.cuda.synchronize()
    
    prefill_time = start_event.elapsed_time(end_event)
    
    # 测试Compute kernel
    start_event.record()
    for _ in range(runs):
        green_context.run_compute_intensive_kernel(input_params, iterations_per_thread)
    end_event.record()
    torch.cuda.synchronize()
    
    compute_time = start_event.elapsed_time(end_event)
    
    total_time = prefill_time + compute_time
    
    print(f"Prefill时间: {prefill_time:.3f} ms")
    print(f"Compute时间: {compute_time:.3f} ms")
    print(f"总时间: {total_time:.3f} ms")
    
    return {"prefill": prefill_time, "compute": compute_time, "total": total_time}

def main():
    parser = argparse.ArgumentParser(description='混合Prefill + 计算密集型Kernel测试')
    parser.add_argument('--mode', choices=['simple', 'hybrid', 'both'], default='simple', 
                       help='测试模式: simple=仅计算kernel, hybrid=混合测试, both=两者都测试')
    parser.add_argument('--sm_prefill', type=int, default=40, help='Prefill使用的SM数量')
    parser.add_argument('--sm_compute', type=int, default=30, help='Compute使用的SM数量')
    parser.add_argument('--batch_size', type=int, default=32, help='Prefill的batch size')
    parser.add_argument('--runs', type=int, default=10, help='运行次数')
    parser.add_argument('--compute_iterations', type=int, default=2000, help='计算kernel的迭代次数')
    args = parser.parse_args()

    print(f"=== 混合Prefill + 计算密集型Kernel测试 ===")
    print(f"测试模式: {args.mode}")
    
    # 输入参数（数据量很小）
    input_params = torch.tensor([1.0, 2.0, 0.5, 3.14], dtype=torch.float32, device='cuda')
    print(f"计算kernel输入数据: {input_params.numel() * 4} bytes (仅 {input_params.numel()} 个float)")
    
    if args.mode in ['simple', 'both']:
        # 简化测试 - 仅测试计算密集型kernel
        print("\n=== 计算密集型Kernel测试 ===")
        
        test_iterations = [500, 1000, 2000]
        
        for iterations in test_iterations:
            print(f"\n--- 测试计算强度: {iterations} ---")
            
            # 简单性能测试
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # 预热
            green_context.run_compute_intensive_kernel(input_params, 10)
            torch.cuda.synchronize()
            
            # 测试
            start_event.record()
            green_context.run_compute_intensive_kernel(input_params, iterations)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed = start_event.elapsed_time(end_event)
            
            # 估算GFLOPS
            total_threads = 64 * 256
            ops_per_iteration = 15
            total_ops = total_threads * iterations * ops_per_iteration
            gflops = (total_ops / 1e9) / (elapsed / 1000)
            
            print(f"执行时间: {elapsed:.3f} ms")
            print(f"计算性能: {gflops:.1f} GFLOPS")
    
    if args.mode in ['hybrid', 'both']:
        # 混合测试 - Prefill + Compute
        print(f"\n=== 混合测试 (计算强度: {args.compute_iterations}) ===")
        
        # 设置数据
        q, kv_cache, kv_indices, kv_indptr, kv_last_len, qo_indptr = setup_paged_data(
            batch_size=args.batch_size, qo_len=4096)
        
        # 数据量对比
        prefill_data = q.numel() * 2 + kv_cache.numel() * 2  # 估算读写量 (bytes)
        compute_data = input_params.numel() * 4  # bytes
        
        print(f"Prefill数据量: ~{prefill_data/1024/1024:.1f} MB (内存密集)")
        print(f"Compute数据量: {compute_data} bytes (计算密集)")
        print(f"数据量比例: 1:{prefill_data//compute_data:.0f}")
        
        # 顺序测试
        sequential_results = test_sequential_hybrid(
            q, kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_len,
            input_params, args.compute_iterations, args.runs)
        
        # 并发测试
        concurrent_results = test_concurrent_hybrid(
            q, kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_len,
            input_params, args.compute_iterations, args.runs, 
            args.sm_prefill, args.sm_compute)
        
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
            prefill_time = concurrent_results.get('prefill', 0)
            compute_time = concurrent_results.get('compute', 0)
            
            if prefill_time > 0 and compute_time > 0:
                print(f"\n--- 工作负载特性 ---")
                print(f"Prefill (内存密集): {prefill_time:.3f} ms")
                print(f"Compute (计算密集): {compute_time:.3f} ms")
                balance = min(prefill_time, compute_time)/max(prefill_time, compute_time)
                print(f"工作负载平衡度: {balance:.2f}")
                
                if balance > 0.8:
                    print("✅ 工作负载平衡良好，重叠效果最佳")
                elif balance > 0.5:
                    print("⚠️  工作负载不够平衡，可调整SM分配或计算强度")
                else:
                    print("❌ 工作负载严重不平衡，建议调整参数")

if __name__ == "__main__":
    main()
