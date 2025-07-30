import torch
import flashinfer
import green_context_lib as green_context
import argparse

page_size = 16

def setup_data(batch_size=64, kv_len=4096, qo_len=4096):
    """设置 GQA prefill 测试数据 (使用 FlashInfer BatchPrefillWithPagedKVCacheWrapper)"""
    num_kv_heads = 8   # GQA: KV heads
    num_qo_heads = 64  # GQA: Query heads
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
    
    # QO indptr: 每个序列的查询范围 (prefill阶段每个序列可能有多个query token)
    qo_indptr = torch.arange(0, batch_size * qo_len + 1, qo_len, dtype=torch.int32, device="cuda:0")

    print(f"=== FlashInfer GQA Prefill 基准测试 ===")
    print(f"Batch size: {batch_size}")
    print(f"Query length: {qo_len}")
    print(f"KV length: {kv_len}")
    print(f"GQA: {num_qo_heads} query heads, {num_kv_heads} KV heads")
    print(f"Page size: {page_size}, Total pages: {total_pages}")
    print(f"Q shape (NHD): {q.shape}")
    print(f"KV cache shape (NHD): {paged_kv_cache.shape}")

    return q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr, num_qo_heads, num_kv_heads, head_dim, page_size

def benchmark_flashinfer(q, paged_kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, qo_indptr, 
                        num_qo_heads, num_kv_heads, head_dim, page_size, sm_count=None, runs=10):
    """使用指定SM数量进行FlashInfer GQA prefill基准测试"""
    print(f"\n--- 测试 {'无限制' if sm_count is None else f'{sm_count} SMs'} ---")

    if sm_count:
        if not green_context.create_green_context(sm_count=sm_count):
             print(f"Failed to create Green Context with {sm_count} SMs")
             return None

    try:
        # 分配workspace buffer (prefill通常需要更大的workspace)
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")  # 128MB workspace
        
        # 创建 BatchPrefillWithPagedKVCacheWrapper
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
        
        # Plan phase
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,  # Prefill 需要causal mask
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
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        print(f"平均时间: {avg_time:.3f} ± {std_dev:.3f} ms")
        return avg_time

    finally:
        if sm_count:
            green_context.destroy_green_context()

def main():
    parser = argparse.ArgumentParser(description='FlashInfer GQA Prefill Benchmark')
    parser.add_argument('--sm', type=int, nargs='+', default=[None, 40, 78, 116, 154, 188],
                        help='List of SM counts to test. "None" for unlimited.')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for prefill testing')
    parser.add_argument('--qo_len', type=int, default=4096,
                        help='Query/Output sequence length for prefill')
    parser.add_argument('--kv_len', type=int, default=4096,
                        help='Key/Value sequence length')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of benchmark runs')
    args = parser.parse_args()

    sm_counts = [count if count is not None else None for count in args.sm]

    data = setup_data(batch_size=args.batch_size, kv_len=args.kv_len, qo_len=args.qo_len)
    
    results = {}
    for sm_count in sm_counts:
        avg_time = benchmark_flashinfer(*data, sm_count=sm_count, runs=args.runs)
        if avg_time:
            results[sm_count] = avg_time

    print(f"\n=== 性能对比结果 ===")
    if None in results:
        baseline = results[None]
        print(f"{'配置':<15} {'时间(ms)':<12} {'vs基线':<10} {'相对性能':<10}")
        print("-" * 50)
        for sm_count, time_ms in results.items():
            config_name = "无限制" if sm_count is None else f"{sm_count} SMs"
            ratio = time_ms / baseline
            perf = baseline / time_ms
            print(f"{config_name:<15} {time_ms:<12.3f} {ratio:.2f}x      {perf*100:.1f}%")

if __name__ == "__main__":
    main() 