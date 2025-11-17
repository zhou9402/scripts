import torch
import flashinfer
import green_context_lib as green_context
import time

page_size = 16
num_kv_heads = 8   # GQA: KV heads
num_qo_heads = 64  # GQA: Query heads
head_dim = 128

# 配置参数
PREFILL_BATCH_SIZE = 4
PREFILL_QO_LEN = 4096
DECODE_BATCH_SIZE = 128
DECODE_QO_LEN = 1
KV_LEN = 4096
RUNS = 10
STEP = 8

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

def run_baseline_tests():
    """在完整SM下分别测试prefill和decode的基线性能 (顺序执行)"""
    print("\n" + "=" * 70)
    print("步骤 1: 测试完整SM下的基线性能 (顺序执行)")
    print("=" * 70)
    print("说明: 两个任务都在完整SM上独立测试，总时间为两者之和")
    
    # 设置数据
    prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = \
        setup_paged_data(batch_size=PREFILL_BATCH_SIZE, qo_len=PREFILL_QO_LEN, kv_len=KV_LEN)
    decode_q, decode_kv_paged, decode_indices, decode_indptr, decode_last_len, decode_qo_indptr = \
        setup_paged_data(batch_size=DECODE_BATCH_SIZE, qo_len=DECODE_QO_LEN, kv_len=KV_LEN)
    
    # 测试 Prefill
    print("\n测试 Prefill (完整SM)...")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    prefill_wrapper.plan(qo_indptr=prefill_qo_indptr, paged_kv_indptr=prefill_indptr, 
                        paged_kv_indices=prefill_indices, paged_kv_last_page_len=prefill_last_len, 
                        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, 
                        head_dim_qk=head_dim, page_size=page_size, causal=True)
    
    # 预热
    for _ in range(3):
        prefill_wrapper.run(prefill_q, prefill_kv_paged)
    torch.cuda.synchronize()
    
    # 测试
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(RUNS):
        prefill_wrapper.run(prefill_q, prefill_kv_paged)
    end.record()
    torch.cuda.synchronize()
    prefill_baseline = start.elapsed_time(end)
    
    # 测试 Decode
    print("测试 Decode (完整SM)...")
    decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
    decode_wrapper.plan(qo_indptr=decode_qo_indptr, paged_kv_indptr=decode_indptr, 
                       paged_kv_indices=decode_indices, paged_kv_last_page_len=decode_last_len, 
                       num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, 
                       head_dim_qk=head_dim, page_size=page_size, causal=False)
    
    # 预热
    for _ in range(3):
        decode_wrapper.run(decode_q, decode_kv_paged)
    torch.cuda.synchronize()
    
    # 测试
    start.record()
    total_decode_runs = RUNS * 10
    for _ in range(total_decode_runs):
        decode_wrapper.run(decode_q, decode_kv_paged)
    end.record()
    torch.cuda.synchronize()
    decode_baseline = start.elapsed_time(end)
    
    sequential_total = prefill_baseline + decode_baseline
    
    print(f"\n基线结果 (都在完整SM上独立测试):")
    print(f"  Prefill 时间: {prefill_baseline:.1f} ms ({RUNS} 次运行)")
    print(f"  Decode 时间: {decode_baseline:.1f} ms ({total_decode_runs} 次运行)")
    print(f"  ----------------------------------------")
    print(f"  顺序执行总时间 = {prefill_baseline:.1f} + {decode_baseline:.1f} = {sequential_total:.1f} ms")
    print("=" * 70)
    
    return prefill_baseline, decode_baseline

def run_hybrid_test(primary_sm_count, max_sm):
    """
    运行混合测试：使用Green Context划分SM
    - Primary partition: 运行 Prefill
    - Remaining partition: 运行 Decode
    """
    remaining_sm_count = max_sm - primary_sm_count
    
    print(f"  创建Green Context (Primary={primary_sm_count} SMs, Remaining={remaining_sm_count} SMs)...")
    
    # 创建Green Context，一次性划分SM
    try:
        primary_stream_handle, remaining_stream_handle, actual_primary_sms, actual_remaining_sms = \
            green_context.create_green_context_and_streams(
                intended_primary_sm_count=primary_sm_count,
                primary_stream_priority=-1,  # 高优先级
                remaining_stream_priority=0   # 正常优先级
            )
        
        print(f"  ✓ Green Context创建成功")
        print(f"    实际分配: Primary={actual_primary_sms} SMs, Remaining={actual_remaining_sms} SMs")
        
        # 将stream handle转换为PyTorch stream对象
        primary_stream = torch.cuda.Stream(stream_ptr=primary_stream_handle)
        remaining_stream = torch.cuda.Stream(stream_ptr=remaining_stream_handle)
        
        # 设置数据
        prefill_q, prefill_kv_paged, prefill_indices, prefill_indptr, prefill_last_len, prefill_qo_indptr = \
            setup_paged_data(batch_size=PREFILL_BATCH_SIZE, qo_len=PREFILL_QO_LEN, kv_len=KV_LEN)
        decode_q, decode_kv_paged, decode_indices, decode_indptr, decode_last_len, decode_qo_indptr = \
            setup_paged_data(batch_size=DECODE_BATCH_SIZE, qo_len=DECODE_QO_LEN, kv_len=KV_LEN)
        
        # 在Primary stream上创建Prefill wrapper
        with torch.cuda.stream(primary_stream):
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
            prefill_wrapper.plan(qo_indptr=prefill_qo_indptr, paged_kv_indptr=prefill_indptr, 
                                paged_kv_indices=prefill_indices, paged_kv_last_page_len=prefill_last_len, 
                                num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, 
                                head_dim_qk=head_dim, page_size=page_size, causal=True)
        
        # 在Remaining stream上创建Decode wrapper
        with torch.cuda.stream(remaining_stream):
            decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"), "NHD")
            decode_wrapper.plan(qo_indptr=decode_qo_indptr, paged_kv_indptr=decode_indptr, 
                               paged_kv_indices=decode_indices, paged_kv_last_page_len=decode_last_len, 
                               num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, 
                               head_dim_qk=head_dim, page_size=page_size, causal=False)
        
        # 预热
        with torch.cuda.stream(primary_stream):
            for _ in range(2):
                prefill_wrapper.run(prefill_q, prefill_kv_paged)
        
        with torch.cuda.stream(remaining_stream):
            for _ in range(2):
                decode_wrapper.run(decode_q, decode_kv_paged)
        
        torch.cuda.synchronize()
        
        # 并发执行测试
        print(f"  开始并发执行...")
        overall_start = time.time()
        
        # 记录开始时间
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        
        # Prefill在primary stream上执行
        with torch.cuda.stream(primary_stream):
            prefill_start.record(primary_stream)
            for _ in range(RUNS):
                prefill_wrapper.run(prefill_q, prefill_kv_paged)
            prefill_end.record(primary_stream)
        
        # Decode在remaining stream上执行
        with torch.cuda.stream(remaining_stream):
            decode_start.record(remaining_stream)
            total_decode_runs = RUNS * 10
            for _ in range(total_decode_runs):
                decode_wrapper.run(decode_q, decode_kv_paged)
            decode_end.record(remaining_stream)
        
        # 等待两个stream完成
        torch.cuda.synchronize()
        overall_end = time.time()
        
        overall_time = (overall_end - overall_start) * 1000  # 转换为ms
        prefill_time = prefill_start.elapsed_time(prefill_end)
        decode_time = decode_start.elapsed_time(decode_end)
        
        # 销毁Green Context
        green_context.destroy_green_context()
        
        return {
            'primary_sm': actual_primary_sms,
            'remaining_sm': actual_remaining_sms,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'overall_time': overall_time
        }
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        try:
            green_context.destroy_green_context()
        except:
            pass
        return None

def main():
    # 获取GPU的实际SM数量
    device_props = torch.cuda.get_device_properties(0)
    MAX_SM = device_props.multi_processor_count
    
    print("\n" + "=" * 70)
    print("Hybrid Prefill + Decode SM 分配测试")
    print("=" * 70)
    print(f"配置:")
    print(f"  GPU总SM数: {MAX_SM}")
    print(f"  Prefill: Batch={PREFILL_BATCH_SIZE}, QO_len={PREFILL_QO_LEN}")
    print(f"  Decode: Batch={DECODE_BATCH_SIZE}, QO_len={DECODE_QO_LEN}")
    print(f"  KV_len={KV_LEN}, 运行{RUNS}次")
    print("=" * 70)
    
    # 先运行基线测试
    prefill_baseline, decode_baseline = run_baseline_tests()
    baseline_sequential = prefill_baseline + decode_baseline
    
    # 生成测试的primary SM数量列表 (primary用于prefill)
    primary_sm_counts = list(range(8, MAX_SM, STEP))
    
    print(f"\n步骤 2: 测试不同SM分配方案")
    print(f"将测试 {len(primary_sm_counts)} 种SM分配方案")
    print(f"Primary (Prefill) SM范围: 8 到 {primary_sm_counts[-1]}")
    print("=" * 70)
    
    all_results = []
    
    # 运行测试
    for i, primary_sm in enumerate(primary_sm_counts, 1):
        remaining_sm = MAX_SM - primary_sm
        print(f"\n[{i}/{len(primary_sm_counts)}] 测试 Primary={primary_sm} SMs (Prefill), Remaining={remaining_sm} SMs (Decode)...")
        
        result = run_hybrid_test(primary_sm, MAX_SM)
        
        if result:
            all_results.append(result)
            print(f"  ✓ 总时间={result['overall_time']:.1f}ms, Prefill={result['prefill_time']:.1f}ms, Decode={result['decode_time']:.1f}ms")
        else:
            print(f"  ✗ 测试失败")
    
    # 输出结果
    print("\n" + "=" * 100)
    print("测试结果汇总")
    print("=" * 100)
    print(f"基线 (完整SM顺序执行): {baseline_sequential:.1f} ms (Prefill={prefill_baseline:.1f}ms + Decode={decode_baseline:.1f}ms)")
    print()
    print(f"{'Prefill_SM':<11} {'Decode_SM':<11} {'Prefill(ms)':<13} {'Decode(ms)':<12} {'总时间(ms)':<12} {'vs基线':<10} {'标记':<10}")
    print("-" * 100)
    
    if all_results:
        # 找到最快的总时间
        best_time = min(r['overall_time'] for r in all_results)
        
        for result in all_results:
            speedup = baseline_sequential / result['overall_time']
            is_best = "★ 最快" if result['overall_time'] == best_time else ""
            print(f"{result['primary_sm']:<11} {result['remaining_sm']:<11} "
                  f"{result['prefill_time']:<13.1f} {result['decode_time']:<12.1f} "
                  f"{result['overall_time']:<12.1f} {speedup:<10.2f}x {is_best:<10}")
        
        print("=" * 100)
        print(f"\n性能总结:")
        best_result = min(all_results, key=lambda x: x['overall_time'])
        speedup = baseline_sequential / best_result['overall_time']
        improvement = (1 - best_result['overall_time'] / baseline_sequential) * 100
        
        print(f"  基线 (顺序执行): {baseline_sequential:.1f} ms")
        print(f"  最优配置时间: {best_result['overall_time']:.1f} ms")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  性能提升: {improvement:.1f}%")
        print(f"\n  最优SM分配:")
        print(f"    Prefill: {best_result['primary_sm']} SMs ({best_result['primary_sm']/MAX_SM*100:.1f}%)")
        print(f"    Decode: {best_result['remaining_sm']} SMs ({best_result['remaining_sm']/MAX_SM*100:.1f}%)")
        print("=" * 100)
    else:
        print("没有成功的测试结果")
        print("=" * 100)

if __name__ == "__main__":
    main()
