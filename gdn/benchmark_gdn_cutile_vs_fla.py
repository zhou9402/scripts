"""
Benchmark GDN: cuTile vs FLA Chunk implementation.

Compares:
- GDN cuTile (fused kernel)
- FLA chunk_gated_delta_rule (Triton)

Usage:
    python benchmark_gdn_cutile_vs_fla.py
    python benchmark_gdn_cutile_vs_fla.py --batch-size 4 --seq-len 16384 --num-heads 32 --head-dim 128
"""

import argparse
import torch
import torch.nn.functional as F

# cuTile imports
import cuda.tile as ct
from fused_gdn import fused_gdn, CHUNK_SIZE

# FLA imports - import directly to avoid transformers dependency
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from fla.ops.utils.cumsum import chunk_local_cumsum
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def prepare_inputs(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16):
    """Prepare input tensors for both implementations."""
    torch.manual_seed(42)
    
    # Standard GDN inputs
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    K = F.normalize(torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device), p=2, dim=-1)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    Beta = torch.rand(batch_size, seq_len, num_heads, dtype=dtype, device=device).sigmoid()
    G = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device))
    
    # Initial state
    h0 = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=device)
    
    return Q, K, V, Beta, G, h0


def prepare_cutile_inputs(Q, K, V, Beta, G, h0, chunk_size=CHUNK_SIZE):
    """Prepare additional inputs needed for cuTile kernel (T matrix)."""
    B, seq_len, H, d = K.shape
    num_chunks = seq_len // chunk_size
    device = K.device
    dtype = K.dtype
    
    # Compute G_cumsum using FLA's cumsum
    G_cumsum = chunk_local_cumsum(G, chunk_size=chunk_size, cu_seqlens=None)
    
    # Compute A (T matrix) = solve_tril of chunk_scaled_dot_kkt
    A = chunk_scaled_dot_kkt_fwd(
        k=K, beta=Beta, g=G_cumsum, cu_seqlens=None, 
        chunk_size=chunk_size, output_dtype=torch.float32
    )
    T = solve_tril(A=A, cu_seqlens=None, output_dtype=dtype)
    
    # Initial state for cuTile (stored as h^T)
    S_init = h0.to(dtype).transpose(-1, -2).contiguous()
    
    return T, G_cumsum, S_init


def benchmark_fla(Q, K, V, Beta, G, h0, num_warmup=10, num_iters=100):
    """Benchmark FLA chunk_gated_delta_rule."""
    scale = K.shape[-1] ** -0.5
    
    # Warmup
    for _ in range(num_warmup):
        O, S = chunk_gated_delta_rule(
            q=Q.clone(), k=K.clone(), v=V.clone(), g=G.clone(),
            beta=Beta.clone(), scale=scale, initial_state=h0.clone(), output_final_state=True
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        O, S = chunk_gated_delta_rule(
            q=Q.clone(), k=K.clone(), v=V.clone(), g=G.clone(),
            beta=Beta.clone(), scale=scale, initial_state=h0.clone(), output_final_state=True
        )
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / num_iters
    return avg_ms, O, S


def benchmark_cutile(Q, K, V, Beta, T, G_cumsum, S_init, num_warmup=10, num_iters=100, chunk_size=CHUNK_SIZE):
    """Benchmark cuTile fused_gdn."""
    # Warmup
    for _ in range(num_warmup):
        S = S_init.clone()
        O = fused_gdn(S=S, T=T, G=G_cumsum, Beta=Beta, V=V, Q=Q, K=K, chunk_size=chunk_size)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        S = S_init.clone()
        O = fused_gdn(S=S, T=T, G=G_cumsum, Beta=Beta, V=V, Q=Q, K=K, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / num_iters
    # S contains final state (stored as h^T)
    S_final = S.transpose(-1, -2).contiguous().float()
    return avg_ms, O, S_final


def benchmark_cutile_e2e(Q, K, V, Beta, G, h0, num_warmup=10, num_iters=100, chunk_size=CHUNK_SIZE):
    """Benchmark cuTile end-to-end (including preprocessing)."""
    dtype = K.dtype
    
    # Warmup
    for _ in range(num_warmup):
        G_cumsum = chunk_local_cumsum(G, chunk_size=chunk_size, cu_seqlens=None)
        A = chunk_scaled_dot_kkt_fwd(k=K, beta=Beta, g=G_cumsum, cu_seqlens=None, 
                                     chunk_size=chunk_size, output_dtype=torch.float32)
        T = solve_tril(A=A, cu_seqlens=None, output_dtype=dtype)
        S = h0.to(dtype).transpose(-1, -2).contiguous()
        O = fused_gdn(S=S, T=T, G=G_cumsum, Beta=Beta, V=V, Q=Q, K=K, chunk_size=chunk_size)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        G_cumsum = chunk_local_cumsum(G, chunk_size=chunk_size, cu_seqlens=None)
        A = chunk_scaled_dot_kkt_fwd(k=K, beta=Beta, g=G_cumsum, cu_seqlens=None, 
                                     chunk_size=chunk_size, output_dtype=torch.float32)
        T = solve_tril(A=A, cu_seqlens=None, output_dtype=dtype)
        S = h0.to(dtype).transpose(-1, -2).contiguous()
        O = fused_gdn(S=S, T=T, G=G_cumsum, Beta=Beta, V=V, Q=Q, K=K, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / num_iters
    return avg_ms


def compare_outputs(O_fla, O_cutile, S_fla, S_cutile, atol=1e-2, rtol=1e-2):
    """Compare outputs between FLA and cuTile."""
    # Output comparison
    o_diff = (O_fla.float() - O_cutile.float()).abs()
    o_max_diff = o_diff.max().item()
    o_mean_diff = o_diff.mean().item()
    o_close = torch.allclose(O_fla.float(), O_cutile.float(), atol=atol, rtol=rtol)
    
    # State comparison
    s_diff = (S_fla.float() - S_cutile.float()).abs()
    s_max_diff = s_diff.max().item()
    s_mean_diff = s_diff.mean().item()
    s_close = torch.allclose(S_fla.float(), S_cutile.float(), atol=atol, rtol=rtol)
    
    return {
        "o_max_diff": o_max_diff,
        "o_mean_diff": o_mean_diff,
        "o_close": o_close,
        "s_max_diff": s_max_diff,
        "s_mean_diff": s_mean_diff,
        "s_close": s_close,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN: cuTile vs FLA")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--verify", action="store_true", help="Verify outputs match")
    args = parser.parse_args()
    
    B, T, H, d = args.batch_size, args.seq_len, args.num_heads, args.head_dim
    chunk_size = args.chunk_size
    device = "cuda"
    
    print(f"{BOLD}{CYAN}=" * 70)
    print(f"GDN Benchmark: cuTile vs FLA")
    print(f"=" * 70 + f"{RESET}")
    print(f"Config: B={B}, T={T}, H={H}, d={d}, chunk_size={chunk_size}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iters}")
    print("-" * 70)
    
    # Prepare inputs
    print(f"\n{YELLOW}Preparing inputs...{RESET}")
    Q, K, V, Beta, G, h0 = prepare_inputs(B, T, H, d, device=device)
    T_mat, G_cumsum, S_init = prepare_cutile_inputs(Q, K, V, Beta, G, h0, chunk_size)
    
    # Memory info
    torch.cuda.reset_peak_memory_stats()
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU Memory: {mem_allocated:.2f} GB allocated")
    
    # Benchmark FLA
    print(f"\n{YELLOW}Benchmarking FLA chunk_gated_delta_rule (Triton)...{RESET}")
    fla_ms, O_fla, S_fla = benchmark_fla(Q, K, V, Beta, G, h0, args.num_warmup, args.num_iters)
    print(f"  {GREEN}FLA Time: {fla_ms:.3f} ms{RESET}")
    
    # Benchmark cuTile (kernel only)
    print(f"\n{YELLOW}Benchmarking cuTile fused_gdn (kernel only)...{RESET}")
    cutile_ms, O_cutile, S_cutile = benchmark_cutile(
        Q, K, V, Beta, T_mat, G_cumsum, S_init, 
        args.num_warmup, args.num_iters, chunk_size
    )
    print(f"  {GREEN}cuTile Kernel Time: {cutile_ms:.3f} ms{RESET}")
    
    # Benchmark cuTile (end-to-end)
    print(f"\n{YELLOW}Benchmarking cuTile fused_gdn (end-to-end with preprocessing)...{RESET}")
    cutile_e2e_ms = benchmark_cutile_e2e(Q, K, V, Beta, G, h0, args.num_warmup, args.num_iters, chunk_size)
    print(f"  {GREEN}cuTile E2E Time: {cutile_e2e_ms:.3f} ms{RESET}")
    
    # Results summary
    print(f"\n{BOLD}{CYAN}=" * 70)
    print(f"Results Summary")
    print(f"=" * 70 + f"{RESET}")
    print(f"{'Implementation':<35} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'FLA chunk_gated_delta_rule':<35} {fla_ms:<15.3f} {'1.00x (baseline)':<15}")
    print(f"{'cuTile fused_gdn (kernel only)':<35} {cutile_ms:<15.3f} {fla_ms/cutile_ms:.2f}x")
    print(f"{'cuTile fused_gdn (E2E)':<35} {cutile_e2e_ms:<15.3f} {fla_ms/cutile_e2e_ms:.2f}x")
    
    # Throughput
    total_tokens = B * T
    fla_tps = total_tokens / (fla_ms / 1000)
    cutile_tps = total_tokens / (cutile_ms / 1000)
    cutile_e2e_tps = total_tokens / (cutile_e2e_ms / 1000)
    
    print(f"\n{'Throughput (tokens/s):':<35}")
    print(f"  FLA:              {fla_tps/1e6:.2f}M tokens/s")
    print(f"  cuTile (kernel):  {cutile_tps/1e6:.2f}M tokens/s")
    print(f"  cuTile (E2E):     {cutile_e2e_tps/1e6:.2f}M tokens/s")
    
    # Verify correctness
    if args.verify:
        print(f"\n{BOLD}{CYAN}=" * 70)
        print(f"Correctness Verification")
        print(f"=" * 70 + f"{RESET}")
        
        results = compare_outputs(O_fla, O_cutile, S_fla, S_cutile)
        
        o_status = f"{GREEN}✓ PASS{RESET}" if results["o_close"] else f"{RED}✗ FAIL{RESET}"
        s_status = f"{GREEN}✓ PASS{RESET}" if results["s_close"] else f"{RED}✗ FAIL{RESET}"
        
        print(f"Output O: {o_status}")
        print(f"  max diff: {results['o_max_diff']:.6f}, mean diff: {results['o_mean_diff']:.6f}")
        print(f"State S:  {s_status}")
        print(f"  max diff: {results['s_max_diff']:.6f}, mean diff: {results['s_mean_diff']:.6f}")
        
        if results["o_close"] and results["s_close"]:
            print(f"\n{GREEN}{BOLD}✓ All outputs match!{RESET}")
        else:
            print(f"\n{RED}{BOLD}✗ Output mismatch detected!{RESET}")
    
    print()


if __name__ == "__main__":
    main()
