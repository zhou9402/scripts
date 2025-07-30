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
    """Prefill worker进程 (从hybrid.py复制并修改)"""
    try:
        # 在进程内部分配资源并plan
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
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
        
        # 预热
        for _ in range(2):
            wrapper.run(q, kv_cache)
        torch.cuda.synchronize()

        ready_event.wait()  # 等待主线程信号
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
        
    except Exception as e:
        print(f"Prefill worker: 错误 - {e}")
        result_queue.put(("prefill", -1))
    finally:
        green_context.destroy_green_context()
        print("Prefill worker: 已销毁Green Context")

def tma_worker(runs, sm_count, ready_event, result_queue):
    """TMA Load Kernel worker进程 - 每次加载32MB数据"""
    try:
        print("TMA worker: 等待开始...")
        if not green_context.create_green_context(sm_count=sm_count):
            print(f"TMA worker: 无法创建Green Context ({sm_count} SMs)")
            result_queue.put(("tma", -1))
            return
        if not green_context.switch_to_green_context():
            print(f"TMA worker: 无法切换到Green Context")
            green_context.destroy_green_context()
            result_queue.put(("tma", -1))
            return
        
        # 创建32MB测试数据
        data_size = 8 * 1024 * 1024  # 32MB = 8M float32
        src_tensor = torch.randn(data_size, dtype=torch.float32, device='cuda')
        print(f"TMA worker: 创建 {data_size * 4 / 1024 / 1024:.1f} MB 测试数据")
        
        # 预热
        for _ in range(2):
            green_context.run_tma_load_kernel(src_tensor)
        torch.cuda.synchronize()

        ready_event.wait()  # 等待主线程信号
        print(f"TMA worker: 开始执行 (在 {sm_count} SMs Green Context中)")
        
        # 计时开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(runs * 100):
            green_context.run_tma_load_kernel(src_tensor)
            if (i + 1) % 10 == 0:
                print(f"TMA worker: 完成 {i+1}/{runs} 次运行...")
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        
        # 计算带宽
        data_gb = data_size * 4 / 1e9
        total_data_gb = data_gb * runs
        bandwidth = (total_data_gb * 2) / (elapsed_time / 1000)  # 读+写
        
        result_queue.put(("tma", elapsed_time))
        print(f"TMA worker: 完成 {runs} 次运行，耗时: {elapsed_time:.3f} ms")
        print(f"TMA worker: 总数据量: {total_data_gb:.2f} GB, 带宽: {bandwidth:.1f} GB/s")
        
    except Exception as e:
        print(f"TMA worker: 错误 - {e}")
        result_queue.put(("tma", -1))
    finally:
        green_context.destroy_green_context()
        print("TMA worker: 已销毁Green Context")

def sequential_test(args):
    """顺序测试：先运行prefill，再运行TMA"""
    print("\n=== 顺序测试模式 ===")
    
    # 1. 设置prefill数据
    print("1. 设置Prefill数据...")
    prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = setup_paged_data(
        batch_size=args.batch_size, qo_len=4096)
    
    # 2. 创建prefill wrapper
    print("2. 创建Prefill wrapper...")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    prefill_wrapper.plan(qo_indptr=prefill_qo_indptr, paged_kv_indptr=prefill_indptr, 
                        paged_kv_indices=prefill_indices, paged_kv_last_page_len=prefill_last_len, 
                        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                        head_dim_qk=head_dim, page_size=page_size, causal=True)
    
    # 3. 创建TMA测试数据
    print("3. 创建TMA测试数据...")
    data_size = 8 * 1024 * 1024  # 32MB = 8M float32
    src_tensor = torch.randn(data_size, dtype=torch.float32, device='cuda')
    
    # 4. 预热
    print("4. 预热...")
    for _ in range(2):
        prefill_wrapper.run(prefill_q, prefill_kv_paged)
        green_context.run_tma_load_kernel(src_tensor)
    torch.cuda.synchronize()
    
    # 5. 测试Prefill
    print("5. 测试Prefill...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(args.runs):
        prefill_wrapper.run(prefill_q, prefill_kv_paged)
    end_event.record()
    torch.cuda.synchronize()
    
    prefill_time = start_event.elapsed_time(end_event)
    print(f"Prefill时间: {prefill_time:.3f} ms")
    
    # 6. 测试TMA
    print("6. 测试TMA...")
    start_event.record()
    for _ in range(args.runs * 1000):
        green_context.run_tma_load_kernel(src_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    tma_time = start_event.elapsed_time(end_event)
    data_gb = data_size * 4 / 1e9
    total_data_gb = data_gb * args.runs
    bandwidth = (total_data_gb * 2) / (tma_time / 1000)
    
    print(f"TMA时间: {tma_time:.3f} ms")
    print(f"TMA带宽: {bandwidth:.1f} GB/s")
    
    total_time = prefill_time + tma_time
    print(f"\n顺序总时间: {total_time:.3f} ms")
    
    return prefill_time, tma_time, total_time

def concurrent_test(args):
    """并发测试：prefill和TMA同时运行"""
    print("\n=== 并发测试模式 ===")
    
    # 1. 设置prefill数据
    prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = setup_paged_data(
        batch_size=args.batch_size, qo_len=4096)
    
    # 2. 创建进程间通信
    ready_event = mp.Event()
    result_queue = mp.Queue()

    # 3. 创建prefill进程
    prefill_args = (
        prefill_q, prefill_kv_paged, args.runs, args.sm_prefill, ready_event, result_queue,
        prefill_qo_indptr, prefill_indptr, prefill_indices, prefill_last_len
    )
    prefill_process = mp.Process(target=prefill_worker, args=prefill_args)

    # 4. 创建TMA进程
    tma_args = (args.runs, args.sm_tma, ready_event, result_queue)
    tma_process = mp.Process(target=tma_worker, args=tma_args)

    # 5. 执行并发测试
    print("开始并发性能测试...")
    
    # 总体计时
    overall_start_event = torch.cuda.Event(enable_timing=True)
    overall_end_event = torch.cuda.Event(enable_timing=True)
    
    overall_start_event.record()
    
    prefill_process.start()
    tma_process.start()
    
    # 等待worker初始化完成
    time.sleep(3)
    
    ready_event.set()  # 发送开始信号
    
    prefill_process.join()
    tma_process.join()
    
    overall_end_event.record()
    torch.cuda.synchronize()
    
    # 6. 收集结果
    results = {}
    for _ in range(2):
        worker_type, elapsed_time = result_queue.get()
        results[worker_type] = elapsed_time
    
    overall_time = overall_start_event.elapsed_time(overall_end_event)
    
    print("\n=== 并发测试完成 ===")
    print(f"并发执行总时间: {overall_time:.3f} ms")
    
    if 'prefill' in results and results['prefill'] > 0:
        print(f"Prefill 时间: {results['prefill']:.3f} ms")
    else:
        print(f"Prefill 时间: 执行失败")
    
    if 'tma' in results and results['tma'] > 0:
        print(f"TMA 时间: {results['tma']:.3f} ms")
    else:
        print(f"TMA 时间: 执行失败")
    
    # 计算重叠效率
    if 'prefill' in results and 'tma' in results and results['prefill'] > 0 and results['tma'] > 0:
        sequential_estimate = results['prefill'] + results['tma']
        overlap_efficiency = (sequential_estimate - overall_time) / sequential_estimate * 100
        print(f"估计顺序执行时间: {sequential_estimate:.3f} ms")
        print(f"重叠效率: {overlap_efficiency:.1f}%")
        
        return results['prefill'], results['tma'], overall_time, overlap_efficiency
    
    return None, None, overall_time, 0

def main():
    parser = argparse.ArgumentParser(description='混合Prefill + TMA Load Kernel测试')
    parser.add_argument('--sm_prefill', type=int, default=40, help='Prefill Green Context的SM数量')
    parser.add_argument('--sm_tma', type=int, default=30, help='TMA Green Context的SM数量')
    parser.add_argument('--batch_size', type=int, default=32, help='Prefill batch size')
    parser.add_argument('--runs', type=int, default=10, help='每个kernel的运行次数')
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'both'], default='both', 
                       help='测试模式: sequential(顺序), concurrent(并发), both(两者)')
    args = parser.parse_args()

    # 为CUDA + multiprocessing 设置 'spawn' 启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"=== 混合Prefill + TMA测试 ===")
    print(f"配置:")
    print(f"  Prefill SM数量: {args.sm_prefill}")
    print(f"  TMA SM数量: {args.sm_tma}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  运行次数: {args.runs}")
    print(f"  TMA数据大小: 32MB per run")

    if args.mode in ['sequential', 'both']:
        seq_prefill, seq_tma, seq_total = sequential_test(args)
    
    if args.mode in ['concurrent', 'both']:
        conc_prefill, conc_tma, conc_total, overlap_eff = concurrent_test(args)
    
    # 如果运行了两种模式，进行对比
    if args.mode == 'both':
        print(f"\n=== 性能对比 ===")
        print(f"顺序模式总时间: {seq_total:.3f} ms")
        print(f"并发模式总时间: {conc_total:.3f} ms")
        if conc_total > 0:
            speedup = seq_total / conc_total
            print(f"并发加速比: {speedup:.2f}x")
            print(f"重叠效率: {overlap_eff:.1f}%")

if __name__ == "__main__":
    main() 