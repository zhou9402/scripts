#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Test KDA Optimized Implementation.

Tests correctness against:
1. FLA naive_recurrent_kda (reference)
2. FLA chunk_kda (optimized reference)

Usage:
    cd /ran/kda_cutile/kda_optimized
    PYTHONPATH=/ran/kda_cutile/flash-linear-attention:$PYTHONPATH python test.py
"""

import argparse
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/ran/kda_cutile/kda_optimized')

from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda

# Import from kda_optimized package
from kda_optimized import kda_forward

# Colors for terminal output
GREEN, RED, YELLOW, CYAN, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"


def compare_tensors(name, t1, t2, atol=1e-2, rtol=1e-2):
    """Compare two tensors and print results."""
    diff = (t1.float() - t2.float()).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    is_close = torch.allclose(t1.float(), t2.float(), atol=atol, rtol=rtol)
    
    total = diff.numel()
    pct_1e2 = 100.0 * (diff < 1e-2).sum().item() / total
    pct_1e3 = 100.0 * (diff < 1e-3).sum().item() / total
    
    status = f"{GREEN}✓{RESET}" if is_close else f"{RED}✗{RESET}"
    print(f"  {status} {name}: max={max_diff:.6f}, mean={mean_diff:.6f}")
    print(f"      |diff|<1e-2: {pct_1e2:.2f}%, |diff|<1e-3: {pct_1e3:.2f}%")
    return is_close


def prepare_inputs(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16):
    """Prepare input tensors."""
    torch.manual_seed(42)
    
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    Beta = torch.rand(batch_size, seq_len, num_heads, dtype=dtype, device=device).sigmoid()
    G = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float, device=device))
    # Clamp G to [-5, 0] to enable safe_gate (TensorCore acceleration)
    G = G.clamp(-5, 0)
    h0 = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=device)
    scale = head_dim ** -0.5
    
    # L2 normalize Q, K
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)
    
    return Q, K, V, Beta, G, h0, scale, Q_norm, K_norm


def test_vs_naive(batch_size=1, seq_len=128, num_heads=2, head_dim=128, chunk_size=64):
    """Test KDA Optimized vs naive_recurrent_kda for correctness."""
    device, dtype = "cuda", torch.bfloat16
    
    print(f"{BOLD}{CYAN}Test: KDA Optimized vs Naive{RESET}")
    print(f"B={batch_size}, T={seq_len}, H={num_heads}, d={head_dim}, chunk_size={chunk_size}")
    print("-" * 60)
    
    Q, K, V, Beta, G, h0, scale, Q_norm, K_norm = prepare_inputs(
        batch_size, seq_len, num_heads, head_dim, device, dtype
    )
    
    # Naive KDA (reference)
    print(f"\n{YELLOW}Running naive_recurrent_kda...{RESET}")
    O_naive, S_naive = naive_recurrent_kda(
        q=Q_norm.clone(), k=K_norm.clone(), v=V.clone(), g=G.clone(),
        beta=Beta.clone(), scale=scale, initial_state=h0.clone(), output_final_state=True
    )
    
    # KDA Optimized
    print(f"{YELLOW}Running KDA Optimized...{RESET}")
    O_opt, S_opt = kda_forward(
        q=Q.clone(), k=K.clone(), v=V.clone(), g=G.clone(),
        beta=Beta.clone(), scale=scale, initial_state=h0.clone(),
        output_final_state=True, chunk_size=chunk_size, normalize_qk=True
    )
    
    # Compare
    print(f"\n{BOLD}Results:{RESET}")
    o_ok = compare_tensors("Output O", O_naive, O_opt)
    s_ok = compare_tensors("State S", S_naive, S_opt)
    
    print()
    if o_ok and s_ok:
        print(f"{GREEN}{BOLD}✓ KDA Optimized matches Naive!{RESET}")
    else:
        print(f"{RED}{BOLD}✗ Mismatch detected{RESET}")
    
    return o_ok and s_ok


def test_vs_fla(batch_size=1, seq_len=128, num_heads=2, head_dim=128, chunk_size=64):
    """Test KDA Optimized vs FLA chunk_kda."""
    device, dtype = "cuda", torch.bfloat16
    
    print(f"{BOLD}{CYAN}Test: KDA Optimized vs FLA chunk_kda{RESET}")
    print(f"B={batch_size}, T={seq_len}, H={num_heads}, d={head_dim}, chunk_size={chunk_size}")
    print("-" * 60)
    
    Q, K, V, Beta, G, h0, scale, Q_norm, K_norm = prepare_inputs(
        batch_size, seq_len, num_heads, head_dim, device, dtype
    )
    
    # FLA chunk_kda (with safe_gate=True for TensorCore acceleration)
    print(f"\n{YELLOW}Running FLA chunk_kda (safe_gate=True)...{RESET}")
    O_fla, S_fla = chunk_kda(
        q=Q_norm.clone(), k=K_norm.clone(), v=V.clone(), g=G.clone(),
        beta=Beta.clone(), scale=scale, initial_state=h0.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True
    )
    
    # KDA Optimized (use pre-normalized Q, K to match FLA)
    print(f"{YELLOW}Running KDA Optimized...{RESET}")
    O_opt, S_opt = kda_forward(
        q=Q_norm.clone(), k=K_norm.clone(), v=V.clone(), g=G.clone(),
        beta=Beta.clone(), scale=scale, initial_state=h0.clone(),
        output_final_state=True, chunk_size=chunk_size, normalize_qk=False
    )
    
    # Compare
    print(f"\n{BOLD}Results:{RESET}")
    o_ok = compare_tensors("Output O", O_fla, O_opt)
    s_ok = compare_tensors("State S", S_fla, S_opt)
    
    print()
    if o_ok and s_ok:
        print(f"{GREEN}{BOLD}✓ KDA Optimized matches FLA chunk_kda!{RESET}")
    else:
        print(f"{RED}{BOLD}✗ Mismatch detected{RESET}")
    
    return o_ok and s_ok


def benchmark(batch_size=1, seq_len=8192, num_heads=96, head_dim=128, 
              chunk_size=64, num_warmup=10, num_iters=100):
    """Benchmark KDA Optimized vs FLA chunk_kda."""
    device, dtype = "cuda", torch.bfloat16
    
    print(f"{BOLD}{CYAN}Benchmark: KDA Optimized vs FLA{RESET}")
    print(f"B={batch_size}, T={seq_len}, H={num_heads}, d={head_dim}, chunk_size={chunk_size}")
    print(f"Warmup: {num_warmup}, Iterations: {num_iters}")
    print("-" * 60)
    
    Q, K, V, Beta, G, h0, scale, Q_norm, K_norm = prepare_inputs(
        batch_size, seq_len, num_heads, head_dim, device, dtype
    )
    
    # FLA Benchmark (with safe_gate=True for TensorCore acceleration)
    print(f"\n{YELLOW}Benchmarking FLA chunk_kda (safe_gate=True)...{RESET}")
    for _ in range(num_warmup):
        O_fla, _ = chunk_kda(q=Q_norm, k=K_norm, v=V, g=G, beta=Beta, scale=scale, 
                            initial_state=h0.clone(), output_final_state=True,
                            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        O_fla, _ = chunk_kda(q=Q_norm, k=K_norm, v=V, g=G, beta=Beta, scale=scale,
                            initial_state=h0.clone(), output_final_state=True,
                            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True)
    end.record()
    torch.cuda.synchronize()
    fla_ms = start.elapsed_time(end) / num_iters
    print(f"  FLA Time: {fla_ms:.3f} ms")
    
    # KDA Optimized Benchmark
    print(f"\n{YELLOW}Benchmarking KDA Optimized...{RESET}")
    for _ in range(num_warmup):
        O_opt, _ = kda_forward(q=Q_norm, k=K_norm, v=V, g=G, beta=Beta, scale=scale,
                              initial_state=h0.clone(), output_final_state=True, 
                              chunk_size=chunk_size, normalize_qk=False)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(num_iters):
        O_opt, _ = kda_forward(q=Q_norm, k=K_norm, v=V, g=G, beta=Beta, scale=scale,
                              initial_state=h0.clone(), output_final_state=True,
                              chunk_size=chunk_size, normalize_qk=False)
    end.record()
    torch.cuda.synchronize()
    opt_ms = start.elapsed_time(end) / num_iters
    print(f"  KDA Optimized Time: {opt_ms:.3f} ms")
    
    # Results
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Results Summary{RESET}")
    print(f"{'='*60}")
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 60)
    print(f"{'FLA chunk_kda':<25} {fla_ms:<15.3f} {'1.00x (baseline)':<15}")
    print(f"{'KDA Optimized':<25} {opt_ms:<15.3f} {fla_ms/opt_ms:.2f}x")
    
    return fla_ms, opt_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KDA Optimized Implementation")
    parser.add_argument("--test", choices=["naive", "fla", "benchmark", "all"], default="all",
                       help="Test to run: naive (vs naive_recurrent), fla (vs chunk_kda), "
                            "benchmark (speed comparison), all (all tests)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=96)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    
    args = parser.parse_args()
    
    print(f"{BOLD}KDA Optimized Test Suite{RESET}\n")
    
    results = []
    
    if args.test in ("naive", "all"):
        print("=" * 70)
        results.append(("vs Naive", test_vs_naive(
            args.batch_size, args.seq_len, args.num_heads, args.head_dim, args.chunk_size
        )))
        print()
    
    if args.test in ("fla", "all"):
        print("=" * 70)
        results.append(("vs FLA", test_vs_fla(
            args.batch_size, args.seq_len, args.num_heads, args.head_dim, args.chunk_size
        )))
        print()
    
    if args.test in ("benchmark", "all"):
        print("=" * 70)
        benchmark(args.batch_size, args.seq_len, args.num_heads, 
                 args.head_dim, args.chunk_size)
        results.append(("Benchmark", True))
        print()
    
    # Summary
    if len(results) > 1:
        print("=" * 70)
        print(f"{BOLD}Summary:{RESET}")
        for name, passed in results:
            status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
            print(f"  {name}: {status}")
