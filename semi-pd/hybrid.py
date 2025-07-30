import torch
import flashinfer
import green_context_lib as green_context
import argparse
import torch.multiprocessing as mp
import time

page_size = 16
num_kv_heads = 8   # GQA: KV heads
num_qo_heads = 64  # GQA: Query heads
head_dim = 128

def setup_paged_data(batch_size=64, kv_len=4096, qo_len=4096):
    """设置 Paged KV Cache 测试数据"""
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
    """在green context下运行prefill的进程函数 (Paged)"""
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

def decode_worker(q, kv_cache, runs, sm_count, ready_event, result_queue,
                    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len):
    """在green context下运行decode的进程函数"""
    # 在进程内部分配资源并plan
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    wrapper.plan(qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr, paged_kv_indices=paged_kv_indices,
                         paged_kv_last_page_len=paged_kv_last_page_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                         head_dim_qk=head_dim, page_size=page_size, causal=False)

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
        
        total_runs = runs * 100  # 运行更多次以确保重叠
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

def sequential_worker(wrapper, data_args, runs, kernel_name, run_multiplier=1):
    """顺序执行指定kernel的函数（批量提交版本）"""
    print(f"\n--- 提交 {kernel_name} 任务 ---")
    
    total_runs = runs * run_multiplier
    
    # 计时开始
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # 批量提交所有任务
    for i in range(total_runs):
        if (i + 1) % 100 == 0:
            print(f"  {kernel_name}: 提交第 {i+1}/{total_runs} 个任务...")
        wrapper.run(*data_args)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"  {kernel_name}: {total_runs} 个任务提交完成，耗时: {elapsed_time:.3f} ms")
    
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description='Concurrent FlashInfer Prefill + Decode Benchmark')
    parser.add_argument('--sm_prefill', type=int, default=40, help='SM count for prefill Green Context (仅在并发模式下有效)')
    parser.add_argument('--sm_decode', type=int, default=30, help='SM count for decode Green Context (仅在并发模式下有效)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--mode', choices=['sequential', 'concurrent'], default='concurrent', help='执行模式')
    args = parser.parse_args()

    # 为CUDA + multiprocessing 设置 'spawn' 启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"=== FlashInfer {args.mode.capitalize()} 测试 ===")
    
    # 1. 设置数据
    prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = setup_paged_data(batch_size=args.batch_size, qo_len=4096)
    decode_q, decode_kv_paged, decode_indices, decode_indptr, decode_last_len, decode_qo_indptr = setup_paged_data(batch_size=args.batch_size, qo_len=1)
    
    if args.mode == 'concurrent':
        # 4. 创建进程
        ready_event = mp.Event()
        result_queue = mp.Queue()

        prefill_args = (
            prefill_q, prefill_kv_paged, args.runs, args.sm_prefill, ready_event, result_queue,
            prefill_qo_indptr, prefill_indptr, prefill_indices, prefill_last_len
        )
        prefill_process = mp.Process(target=prefill_worker, args=prefill_args)

        decode_args = (
            decode_q, decode_kv_paged, args.runs, args.sm_decode, ready_event, result_queue,
            decode_qo_indptr, decode_indptr, decode_indices, decode_last_len
        )
        decode_process = mp.Process(target=decode_worker, args=decode_args)

        # 5. 计时并执行
        print("\n开始并发性能测试...")
        
        # 总体计时开始
        overall_start_event = torch.cuda.Event(enable_timing=True)
        overall_end_event = torch.cuda.Event(enable_timing=True)
        
        overall_start_event.record()
        
        prefill_process.start()
        decode_process.start()
        
        # 等待worker初始化完成
        time.sleep(3) 
        
        ready_event.set() # 发送开始信号
        
        prefill_process.join()
        decode_process.join()
        
        overall_end_event.record()
        torch.cuda.synchronize()
        
        # 收集结果
        results = {}
        for _ in range(2):  # 收集两个进程的结果
            worker_type, elapsed_time = result_queue.get()
            results[worker_type] = elapsed_time
        
        overall_time = overall_start_event.elapsed_time(overall_end_event)
        
        print("\n=== 并发测试完成 ===")
        print(f"并发执行总时间: {overall_time:.3f} ms")
        if 'prefill' in results and results['prefill'] > 0:
            print(f"Prefill 时间: {results['prefill']:.3f} ms")
        else:
            print(f"Prefill 时间: 执行失败")
        if 'decode' in results and results['decode'] > 0:
            print(f"Decode 时间: {results['decode']:.3f} ms")
        else:
            print(f"Decode 时间: 执行失败")
        
        # 计算重叠效率
        if 'prefill' in results and 'decode' in results and results['prefill'] > 0 and results['decode'] > 0:
            sequential_estimate = results['prefill'] + results['decode']
            overlap_efficiency = (sequential_estimate - overall_time) / sequential_estimate * 100
            print(f"估计顺序执行时间: {sequential_estimate:.3f} ms")
            print(f"重叠效率: {overlap_efficiency:.1f}%")
    else: # sequential 模式
        # 顺序模式下，wrapper需要在主进程中创建
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
        decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    
        prefill_wrapper.plan(qo_indptr=prefill_qo_indptr, paged_kv_indptr=prefill_indptr, paged_kv_indices=prefill_indices,
                             paged_kv_last_page_len=prefill_last_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                             head_dim_qk=head_dim, page_size=page_size, causal=True)
                             
        decode_wrapper.plan(qo_indptr=decode_qo_indptr, paged_kv_indptr=decode_indptr, paged_kv_indices=decode_indices,
                             paged_kv_last_page_len=decode_last_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                             head_dim_qk=head_dim, page_size=page_size, causal=False)
        # 预热
        print("\n预热阶段 (顺序)...")
        for _ in range(2):
            prefill_wrapper.run(prefill_q, prefill_kv_paged)
            decode_wrapper.run(decode_q, decode_kv_paged)
        torch.cuda.synchronize()

        print("\n开始顺序性能测试...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        
        # 批量提交所有prefill任务
        prefill_time = sequential_worker(prefill_wrapper, (prefill_q, prefill_kv_paged), args.runs, "Prefill (Paged)")
        
        # 批量提交所有decode任务
        decode_time = sequential_worker(decode_wrapper, (decode_q, decode_kv_paged), args.runs * 100, "Decode (Paged)")
        
        # 最后一次总同步
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        total_time = start_event.elapsed_time(end_event)
        print("\n=== 顺序执行总结 ===")
        print(f"顺序执行总时间: {total_time:.3f} ms")
        print(f"Prefill 总时间: {prefill_time:.3f} ms")
        print(f"Decode 总时间: {decode_time:.3f} ms")

if __name__ == "__main__":
    main()