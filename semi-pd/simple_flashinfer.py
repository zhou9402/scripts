import torch
import flashinfer
import os
import green_context

kv_len = 4096
num_kv_heads = 8
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention

num_qo_heads = 64
# q = torch.randn(num_qo_heads, head_dim).half().to(0)

# o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
# o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

# # append attention
# append_qo_len = 128
# q = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to(0) # append attention, the last 128 tokens in the KV-Cache are the new tokens
# o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True) # append attention without RoPE on-the-fly, apply causal mask
# o_rope_on_the_fly = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA") # append attention with LLaMA style RoPE on-the-fly, apply causal mask

# prefill attention
qo_len = 8
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0) # prefill attention
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

def benchmark_flashinfer(name="Normal"):
    # warm up
    for _ in range(5):
        _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    torch.cuda.synchronize()

    num_runs = 10
    elapsed_times = []
    for _ in range(num_runs):
        start_event.record()
        o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time_ms)

    avg_time = sum(elapsed_times) / len(elapsed_times)
    print(f"{name} single_prefill_with_kv_cache 平均耗时: {avg_time:.3f} ms")
    return avg_time

# 测试正常模式
print("=== FlashInfer Green Context 测试 ===")
normal_time = benchmark_flashinfer("正常模式")

sm_count = 20
# 测试 Green Context 模式（限制 sm_count 个 SM）
print(f"\n使用 Green Context ({sm_count} SMs)...")
with green_context.GreenContextManager(sm_count=sm_count):
    green_time = benchmark_flashinfer(f"Green Context ({sm_count} SMs)")

# 对比结果
ratio = green_time / normal_time
print(f"\n=== 性能对比 ===")
print(f"正常模式: {normal_time:.3f} ms")
print(f"Green Context ({sm_count} SMs): {green_time:.3f} ms")
print(f"性能比值 (Green/Normal): {ratio:.3f}")