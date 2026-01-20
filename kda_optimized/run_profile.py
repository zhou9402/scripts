#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Profile KDA Optimized vs FLA for nsys analysis.

Usage:
    cd /ran/kda_cutile/kda_optimized
    PYTHONPATH=/ran/kda_cutile/flash-linear-attention:$PYTHONPATH \
        nsys profile -o kda_opt_profile --force-overwrite true python profile.py
    
    # View kernel summary
    nsys stats kda_opt_profile.nsys-rep --report cuda_gpu_kern_sum
    
    # View NVTX timing
    nsys stats kda_opt_profile.nsys-rep --report nvtx_sum
"""

import argparse
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/ran/kda_cutile/kda_optimized')

from fla.ops.kda import chunk_kda
from kda_optimized import kda_forward


def main(batch_size=1, seq_len=8192, num_heads=96, head_dim=128, 
         chunk_size=64, num_iters=10):
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"Config: B={batch_size}, T={seq_len}, H={num_heads}, d={head_dim}, chunk_size={chunk_size}")
    
    # Prepare inputs
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
    
    # ========== Warmup (not profiled) ==========
    print("Warmup...")
    for _ in range(5):
        O_fla, _ = chunk_kda(
            q=Q_norm.clone(), k=K_norm.clone(), v=V.clone(), g=G.clone(),
            beta=Beta.clone(), scale=scale, initial_state=h0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True
        )
        O_opt, _ = kda_forward(
            q=Q_norm.clone(), k=K_norm.clone(), v=V.clone(), g=G.clone(),
            beta=Beta.clone(), scale=scale, initial_state=h0.clone(),
            output_final_state=True, chunk_size=chunk_size, normalize_qk=False
        )
    torch.cuda.synchronize()
    
    # Start profiling
    torch.cuda.cudart().cudaProfilerStart()
    
    # ========== Profile FLA (safe_gate=True) ==========
    print(f"Profiling FLA chunk_kda safe_gate=True ({num_iters} iterations)...")
    torch.cuda.nvtx.range_push("FLA_chunk_kda_safe_gate")
    
    for i in range(num_iters):
        torch.cuda.nvtx.range_push(f"FLA_iter_{i}")
        O_fla, _ = chunk_kda(
            q=Q_norm, k=K_norm, v=V, g=G,
            beta=Beta, scale=scale, initial_state=h0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True
        )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    
    # ========== Profile KDA Optimized ==========
    print(f"Profiling KDA Optimized ({num_iters} iterations)...")
    torch.cuda.nvtx.range_push("KDA_Optimized")
    
    for i in range(num_iters):
        torch.cuda.nvtx.range_push(f"KDA_Opt_iter_{i}")
        O_opt, _ = kda_forward(
            q=Q_norm, k=K_norm, v=V, g=G,
            beta=Beta, scale=scale, initial_state=h0.clone(),
            output_final_state=True, chunk_size=chunk_size, normalize_qk=False
        )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    
    # Stop profiling
    torch.cuda.cudart().cudaProfilerStop()
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile KDA Optimized")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=96)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=10)
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        num_iters=args.num_iters,
    )
