import torch
import flashinfer
import green_context_lib as green_context

page_size = 16

def setup_data(batch_size=64, kv_len=4096):
    """
    设置 GQA decode 测试数据 (使用 FlashInfer BatchPrefillWithPagedKVCacheWrapper)
    GQA配置: 64 query heads : 8 KV heads = 8:1 (每8个query heads共享1个KV head)
    Decode阶段: 每个序列生成1个token
    """
    qo_len = 1   # Decode 阶段每次生成1个token
    num_kv_heads = 8   # GQA: KV heads
    num_qo_heads = 64  # GQA: Query heads (8:1 ratio)
    head_dim = 128

    # Query: flatten format [total_q_len, num_qo_heads, head_dim] -> NHD layout
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
    
    # QO indptr: 每个序列的查询范围 (decode阶段每个序列生成1个token)
    qo_indptr = torch.arange(0, batch_size * qo_len + 1, qo_len, dtype=torch.int32, device="cuda:0")

    return q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr, num_qo_heads, num_kv_heads, head_dim, page_size

def benchmark_flashinfer(q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr, 
                        num_qo_heads, num_kv_heads, head_dim, page_size, sm_count=None, runs=10):
    """使用指定SM数量进行FlashInfer GQA decode基准测试"""

    primary_stream = None
    
    if sm_count:
        try:
            # 使用Green Context的primary partition来限制SM
            primary_stream_handle, remaining_stream_handle, actual_primary_sms, actual_remaining_sms = \
                green_context.create_green_context_and_streams(
                    intended_primary_sm_count=sm_count,
                    primary_stream_priority=0,
                    remaining_stream_priority=0
                )
            primary_stream = torch.cuda.Stream(stream_ptr=primary_stream_handle)
        except Exception as e:
            return None

    try:
        if sm_count:
            # 在primary stream上执行
            with torch.cuda.stream(primary_stream):
                workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
                
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len=paged_kv_last_page_len,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=page_size,
                    causal=False,
                )

                # 预热
                for _ in range(3):
                    _ = wrapper.run(q, paged_kv_cache)
                primary_stream.synchronize()

                # 性能测试
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(runs):
                    start_event.record(primary_stream)
                    _ = wrapper.run(q, paged_kv_cache)
                    end_event.record(primary_stream)
                    primary_stream.synchronize()
                    times.append(start_event.elapsed_time(end_event))
        else:
            # 使用默认stream (完整SM)
            workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
            
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=False,
            )

            # 预热
            for _ in range(3):
                _ = wrapper.run(q, paged_kv_cache)
            torch.cuda.synchronize()

            # 性能测试
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(runs):
                start_event.record()
                _ = wrapper.run(q, paged_kv_cache)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        avg_time = sum(times) / len(times)
        return avg_time

    finally:
        if sm_count:
            green_context.destroy_green_context()

def main():
    # 获取GPU的实际SM数量
    device_props = torch.cuda.get_device_properties(0)
    MAX_SM = device_props.multi_processor_count
    
    # 配置参数
    BATCH_SIZE = 128  # Decode 通常使用较大的 batch size
    KV_LEN = 4096
    RUNS = 10
    STEP = 8
    
    # 生成SM数量列表：从8开始，步长为STEP，直到MAX_SM
    sm_counts = list(range(8, MAX_SM + 1, STEP))
    if MAX_SM not in sm_counts:
        sm_counts.append(MAX_SM)

    print("\n" + "=" * 70)
    print("GQA Decode SM 缩放测试")
    print("=" * 70)
    print(f"配置: Batch={BATCH_SIZE}, KV_len={KV_LEN}, Query_len=1 (decode)")
    print(f"GPU总SM数: {MAX_SM}, 测试范围: 8~{MAX_SM} (步长{STEP}), 运行{RUNS}次")
    print("=" * 70)
    
    data = setup_data(batch_size=BATCH_SIZE, kv_len=KV_LEN)
    
    results = {}
    
    # 首先测试完整SM基线
    print(f"\n[1/{len(sm_counts)+1}] 测试完整SM (无限制)...")
    avg_time = benchmark_flashinfer(*data, sm_count=None, runs=RUNS)
    if avg_time:
        results[None] = avg_time
        print(f"  ✓ 完整SM时间: {avg_time:.3f} ms")
    else:
        print(f"  ✗ 测试失败")
        return
    
    # 测试各个SM数量
    for i, sm_count in enumerate(sm_counts, 2):
        print(f"[{i}/{len(sm_counts)+1}] 测试 {sm_count} SMs...", end="", flush=True)
        avg_time = benchmark_flashinfer(*data, sm_count=sm_count, runs=RUNS)
        if avg_time:
            results[sm_count] = avg_time
            print(f"\r[{i}/{len(sm_counts)+1}] 测试 {sm_count} SMs - ✓ {avg_time:.3f} ms" + " " * 20)
        else:
            print(f"\r[{i}/{len(sm_counts)+1}] 测试 {sm_count} SMs - ✗ 失败" + " " * 20)

    # 输出结果
    baseline = results[None]
    print("\n" + "=" * 70)
    print("性能对比结果")
    print("=" * 70)
    print(f"GPU总SM数: {MAX_SM}")
    print(f"完整SM基线时间: {baseline:.3f} ms\n")
    print(f"{'SM数量':<10} {'占比%':<12} {'时间(ms)':<12} {'性能达到%':<12}")
    print("-" * 70)
    
    # 显示完整SM
    print(f"{MAX_SM:<10} {'100.0%':<12} {baseline:<12.3f} {'100.0%':<12}")
    
    # 显示各个受限SM的结果
    for sm_count in sm_counts:
        if sm_count in results:
            time_ms = results[sm_count]
            sm_percentage = (sm_count / MAX_SM) * 100  # SM占总SM的百分比
            perf_percentage = (baseline / time_ms) * 100  # 性能达到的百分比（完整时间/当前时间）
            print(f"{sm_count:<10} {sm_percentage:<12.1f}% {time_ms:<12.3f} {perf_percentage:<12.1f}%")
    
    print("=" * 70)
    print("\n说明:")
    print("  SM数量: 使用的SM数量")
    print("  占比%: SM数量占GPU总SM的百分比")
    print("  时间(ms): 平均执行时间 (越低越好)")
    print("  性能达到%: 达到完整SM性能的百分比 (越高越好)")
    print("=" * 70)

if __name__ == "__main__":
    main()