import torch
import flashinfer
import green_context_simple as gc
import argparse
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


def concurrent_execution_with_streams(prefill_wrapper, decode_wrapper, 
                                    prefill_data, decode_data, 
                                    primary_stream, remaining_stream, 
                                    runs):
    """使用不同stream并发执行prefill和decode"""
    print("\n=== 开始并发Stream执行 ===")
    
    # 预热两个stream
    print("预热阶段...")
    with torch.cuda.stream(primary_stream):
        for _ in range(3):
            prefill_wrapper.run(*prefill_data)
        primary_stream.synchronize()
    
    with torch.cuda.stream(remaining_stream):
        for _ in range(3):
            decode_wrapper.run(*decode_data)
        remaining_stream.synchronize()
    
    # 准备事件
    overall_start = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    
    # 开始并发执行
    print("开始并发执行...")
    overall_start.record()
    
    # 在primary stream中运行prefill
    with torch.cuda.stream(primary_stream):
        prefill_start.record(primary_stream)
        for i in range(runs):
            if (i + 1) % 5 == 0:
                print(f"  Prefill (Primary): 第 {i+1}/{runs} 次")
            prefill_wrapper.run(*prefill_data)
        prefill_end.record(primary_stream)
    
    # 在remaining stream中运行decode (更多次数)
    decode_runs = runs * 50  # decode通常更快，运行更多次
    with torch.cuda.stream(remaining_stream):
        decode_start.record(remaining_stream)
        for i in range(decode_runs):
            if (i + 1) % 100 == 0:
                print(f"  Decode (Remaining): 第 {i+1}/{decode_runs} 次")
            decode_wrapper.run(*decode_data)
        decode_end.record(remaining_stream)
    
    # 等待所有操作完成
    primary_stream.synchronize()
    remaining_stream.synchronize()
    overall_end.record()
    torch.cuda.synchronize()
    
    # 获取时间
    overall_time = overall_start.elapsed_time(overall_end)
    prefill_time = prefill_start.elapsed_time(prefill_end)
    decode_time = decode_start.elapsed_time(decode_end)
    
    return overall_time, prefill_time, decode_time, decode_runs

def sequential_execution(prefill_wrapper, decode_wrapper, 
                        prefill_data, decode_data, runs):
    """顺序执行作为基准"""
    print("\n=== 开始顺序执行基准测试 ===")
    
    # 预热
    for _ in range(3):
        prefill_wrapper.run(*prefill_data)
        decode_wrapper.run(*decode_data)
    torch.cuda.synchronize()
    
    # 顺序执行prefill
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(runs):
        if (i + 1) % 5 == 0:
            print(f"  顺序Prefill: 第 {i+1}/{runs} 次")
        prefill_wrapper.run(*prefill_data)
    end_event.record()
    torch.cuda.synchronize()
    
    prefill_sequential_time = start_event.elapsed_time(end_event)
    
    # 顺序执行decode
    decode_runs = runs * 50
    start_event.record()
    for i in range(decode_runs):
        if (i + 1) % 100 == 0:
            print(f"  顺序Decode: 第 {i+1}/{decode_runs} 次")
        decode_wrapper.run(*decode_data)
    end_event.record()
    torch.cuda.synchronize()
    
    decode_sequential_time = start_event.elapsed_time(end_event)
    total_sequential_time = prefill_sequential_time + decode_sequential_time
    
    return total_sequential_time, prefill_sequential_time, decode_sequential_time

def main():
    parser = argparse.ArgumentParser(description='Concurrent FlashInfer with Green Context Streams')
    parser.add_argument('--sm_prefill', type=int, default=152, help='SM count for prefill (primary partition)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--runs', type=int, default=10, help='Number of prefill runs')
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'both'], default='both', help='执行模式')
    args = parser.parse_args()

    print(f"=== FlashInfer Green Context Stream 并发测试 ===")
    print(f"模式: {args.mode}")
    print(f"Prefill SM数量: {args.sm_prefill}")
    print(f"批次大小: {args.batch_size}")
    print(f"运行次数: {args.runs}")
    
    try:
        # 1. 创建Green Context Manager
        print(f"\n创建Green Context Manager...")
        manager = gc.GreenContextManager(device_id=0)
        
        # 2. 创建Green Context和streams
        print(f"创建Green Context (Primary: {args.sm_prefill} SMs)...")
        primary_stream_raw, remaining_stream_raw = manager.create_green_context_and_streams(
            intended_primary_partition_sm_count=args.sm_prefill,
            primary_stream_priority=-1,  # 高优先级给prefill
            remaining_stream_priority=0   # 正常优先级给decode
        )
        
        # 获取实际SM分配
        primary_sms, remaining_sms = manager.get_sm_counts()
        print(f"实际SM分配: Primary={primary_sms}, Remaining={remaining_sms}")
        
        # 3. 转换为PyTorch streams
        primary_stream = torch.cuda.Stream(stream_ptr=primary_stream_raw)
        remaining_stream = torch.cuda.Stream(stream_ptr=remaining_stream_raw)
        print(f"PyTorch Streams创建成功")
        print(f"  Primary stream: 0x{primary_stream_raw:x}")
        print(f"  Remaining stream: 0x{remaining_stream_raw:x}")
        
        # 4. 设置数据
        print(f"\n设置测试数据...")
        prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = setup_paged_data(
            batch_size=args.batch_size, qo_len=4096)
        decode_q, decode_kv_paged, decode_indices, decode_indptr, decode_last_len, decode_qo_indptr = setup_paged_data(
            batch_size=args.batch_size, qo_len=1)
        
        # 5. 创建和配置wrappers
        print(f"配置FlashInfer wrappers...")
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
        decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
        
        # 配置prefill wrapper
        prefill_wrapper.plan(
            qo_indptr=prefill_qo_indptr, paged_kv_indptr=prefill_indptr, paged_kv_indices=prefill_indices,
            paged_kv_last_page_len=prefill_last_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, page_size=page_size, causal=True)
        
        # 配置decode wrapper
        decode_wrapper.plan(
            qo_indptr=decode_qo_indptr, paged_kv_indptr=decode_indptr, paged_kv_indices=decode_indices,
            paged_kv_last_page_len=decode_last_len, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, page_size=page_size, causal=False)
        
        prefill_data = (prefill_q, prefill_kv_paged)
        decode_data = (decode_q, decode_kv_paged)
        
        # 6. 执行测试
        if args.mode in ['sequential', 'both']:
            seq_total, seq_prefill, seq_decode = sequential_execution(
                prefill_wrapper, decode_wrapper, prefill_data, decode_data, args.runs)
        
        if args.mode in ['concurrent', 'both']:
            conc_total, conc_prefill, conc_decode, decode_runs = concurrent_execution_with_streams(
                prefill_wrapper, decode_wrapper, prefill_data, decode_data, 
                primary_stream, remaining_stream, args.runs)
        
        # 7. 结果分析
        print(f"\n" + "="*60)
        print(f"性能测试结果")
        print(f"="*60)
        
        if args.mode in ['sequential', 'both']:
            print(f"顺序执行:")
            print(f"  Prefill 时间: {seq_prefill:.3f} ms ({args.runs} 次)")
            print(f"  Decode 时间: {seq_decode:.3f} ms ({args.runs * 50} 次)")
            print(f"  总时间: {seq_total:.3f} ms")
        
        if args.mode in ['concurrent', 'both']:
            print(f"并发执行 (Green Context Streams):")
            print(f"  Primary Stream (Prefill): {conc_prefill:.3f} ms ({args.runs} 次, {primary_sms} SMs)")
            print(f"  Remaining Stream (Decode): {conc_decode:.3f} ms ({decode_runs} 次, {remaining_sms} SMs)")
            print(f"  并发总时间: {conc_total:.3f} ms")
        
        if args.mode == 'both':
            print(f"\n性能比较:")
            speedup = seq_total / conc_total
            overlap_efficiency = (seq_total - conc_total) / seq_total * 100
            print(f"  加速比: {speedup:.2f}x")
            print(f"  重叠效率: {overlap_efficiency:.1f}%")
            print(f"  时间节省: {seq_total - conc_total:.3f} ms")
            
            # SM效率分析
            total_sms = primary_sms + remaining_sms
            primary_efficiency = primary_sms / total_sms
            remaining_efficiency = remaining_sms / total_sms
            print(f"\nSM分配效率:")
            print(f"  Primary partition: {primary_sms}/{total_sms} ({primary_efficiency:.1%})")
            print(f"  Remaining partition: {remaining_sms}/{total_sms} ({remaining_efficiency:.1%})")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        try:
            manager.destroy_streams()
            print(f"\n✅ Green Context资源已清理")
        except:
            pass

if __name__ == "__main__":
    main()