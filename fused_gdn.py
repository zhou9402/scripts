# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fused Gated Delta Net: Fuses scale_vk and gdn kernels into a single kernel

Layout: (B, T, H, D) - Batch, Time/Sequence, Head, Dimension
        No reshape needed - kernel directly loads strided data

Algorithm (fully fused):
  For each chunk:
    # Scale T instead of V/K (mathematically equivalent, more efficient)
    # T @ (v * beta) = (T * beta^T) @ v
    # T @ (k * beta * exp(g)) = (T * beta^T * exp(g)^T) @ k
    T_v = T * beta^T              # [chunk, chunk], scale columns by beta
    T_k = T_v * exp(g)^T          # [chunk, chunk], scale columns by beta * exp(g)
    
    # GDN computation (using original K, V)
    W = T_k @ K                   # [chunk, d]
    U = T_v @ V                   # [chunk, d]
    delta = (U - W @ S^T) * exp(g_last - g)   # [chunk, d], gated delta
    O = (Q @ S^T) * exp(g) + mask(Q @ K^T * exp(g[:, None] - g[None, :])) @ delta
    S = S * exp(g_last) + delta^T @ K   # [d, d], state update with decay

Inputs:
  S: (batch, num_heads, d, d) float32 - state matrices (initialized to 0)
  T: (batch, seq_len, num_heads, chunk_size) bfloat16 - precomputed T
  G: (batch, seq_len, num_heads) bfloat16 - gate values
  Beta: (batch, seq_len, num_heads) bfloat16 - beta values
  V: (batch, seq_len, num_heads, d) bfloat16 - values
  Q: (batch, seq_len, num_heads, d) bfloat16 - queries
  K: (batch, seq_len, num_heads, d) bfloat16 - keys
  
Outputs:
  O: (batch, seq_len, num_heads, d) bfloat16 - output
  S: (batch, num_heads, d, d) float32 - updated states (in-place)
"""

import argparse
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# Default configuration
HEAD_DIM = 128    # d
CHUNK_SIZE = 64   # chunk size for processing


@ct.kernel
def fused_gdn_kernel(
    S,      # (B, H, d, d) float32 - state
    T_in,   # (B, T, H, chunk_size) bfloat16 - precomputed T matrix
    V,      # (B, T, H, d) bfloat16 - values (not pre-scaled)
    Q,      # (B, T, H, d) bfloat16 - queries
    K,      # (B, T, H, d) bfloat16 - keys (not pre-scaled)
    G,      # (B, T, H) bfloat16 - gate values
    Beta,   # (B, T, H) bfloat16 - beta values
    O,      # (B, T, H, d) bfloat16 - output
    d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    num_chunks: ConstInt,  # number of chunks to process
    num_heads: ConstInt,   # H
    seq_len: ConstInt      # T
):
    """
    Fully fused GDN kernel: combines scale_vk and gdn into single kernel.
    
    For each chunk:
      1. Load V, K, Q, G, Beta for chunk
      2. Compute v_scaled = v * beta, k_scaled = k * beta * exp(g) on-the-fly
      3. Compute W = T @ k_scaled, U = T @ v_scaled
      4. Compute delta, O, and update S
    
    Grid: (B * H,) - one block per (b, h) pair.
    """
    linear_idx = ct.bid(0)
    b_idx = linear_idx // num_heads
    h_idx = linear_idx % num_heads
    zero_pad = ct.PaddingMode.ZERO
    
    # Load initial state S
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.reshape(s, (d, d))
    
    # Create causal mask (lower triangular) inside kernel
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]  # [chunk_size, 1]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]  # [1, chunk_size]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)  # [chunk_size, chunk_size] float32
    
    for c in range(num_chunks):
        t_start = c * chunk_size
        
        # Compute actual chunk end (handle last chunk if not divisible)
        t_end = min(t_start + chunk_size, seq_len)
        
        # ============================================================
        # Load raw inputs for this chunk
        # ============================================================
        
        # T: (B, seq_len, H, chunk_size) -> (chunk_size, chunk_size)
        T = ct.load(T_in, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, chunk_size), 
                    padding_mode=zero_pad)
        T = ct.reshape(T, (chunk_size, chunk_size))
        
        # K: (B, T, H, d) -> (chunk_size, d)
        k = ct.load(K, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        k = ct.reshape(k, (chunk_size, d))
        
        # V: (B, T, H, d) -> (chunk_size, d)
        v = ct.load(V, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        v = ct.reshape(v, (chunk_size, d))
        
        # Q: (B, T, H, d) -> (chunk_size, d)
        q = ct.load(Q, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        q = ct.reshape(q, (chunk_size, d))
        
        # G: (B, T, H) -> (chunk_size, 1)
        g_raw = ct.load(G, index=(b_idx, t_start, h_idx), shape=(1, chunk_size, 1), 
                        padding_mode=zero_pad)
        g_raw = ct.reshape(g_raw, (chunk_size, 1))
        
        # Beta: (B, T, H) -> (1, chunk_size) for column-wise scaling of T
        beta = ct.load(Beta, index=(b_idx, t_start, h_idx), shape=(1, chunk_size, 1), 
                       padding_mode=zero_pad)
        beta = ct.reshape(beta, (1, chunk_size))
        
        # Get last g value in chunk for state gating
        g_chunk_last = ct.load(G, index=(b_idx, t_end - 1, h_idx), shape=(1, 1, 1), 
                               padding_mode=zero_pad)
        g_chunk_last = ct.reshape(g_chunk_last, (1, 1))
        
        # ============================================================
        # Compute gating factors (all in bfloat16)
        # ============================================================
        
        # exp(g) for T_k scaling
        exp_g = ct.exp(g_raw)  # [chunk_size, 1] bfloat16
        
        # g_chunk = exp(g_chunk_last - g_raw) for delta scaling
        g_chunk = ct.exp(g_chunk_last - g_raw)  # [chunk_size, 1]
        
        # g_chunk_last_exp = exp(g_chunk_last) for state decay
        g_chunk_last_exp = ct.exp(g_chunk_last)  # [1, 1] scalar
        
        # g_attn_matrix = exp(g_raw[:, None] - g_raw[None, :]) for attention
        g_raw_t = ct.transpose(g_raw)  # [1, chunk_size]
        g_attn_matrix = ct.exp(g_raw - g_raw_t)  # [chunk_size, chunk_size]
        
        # g_out = exp(g_raw) for output scaling (same as exp_g)
        g_out = exp_g  # [chunk_size, 1]
        
        # ============================================================
        # Scale T instead of V and K (mathematically equivalent, more efficient)
        # T @ (v * beta) = (T * beta) @ v  (beta is 1 x chunk_size, broadcasts to columns)
        # T @ (k * beta * exp(g)) = (T * beta * exp(g)^T) @ k
        # ============================================================
        
        # Transpose exp_g for column-wise scaling of T
        exp_g_t = ct.transpose(exp_g)  # [1, chunk_size]
        
        # T_v = T * beta (scale columns by beta, beta is already [1, chunk_size])
        T_v = T * beta  # [chunk_size, chunk_size] bfloat16
        
        # T_k = T_v * exp(g)^T = T * beta * exp(g)^T (scale columns by beta * exp(g))
        T_k = T_v * exp_g_t  # [chunk_size, chunk_size] bfloat16
        
        # ============================================================
        # Compute W = T_k @ K, U = T_v @ V (using original K, V)
        # ============================================================
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        u = ct.astype(ct.mma(T_v, v, acc), K.dtype)
        
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w = ct.astype(ct.mma(T_k, k, acc), K.dtype)
        
        # ============================================================
        # Compute delta = (U - W @ S^T) * g_chunk
        # ============================================================
        
        s_fp16 = ct.astype(s, K.dtype)
        s_t = ct.transpose(s_fp16)
        
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.astype(ct.mma(w, s_t, acc_ws), K.dtype)  # bfloat16
        
        delta = (u - w_st) * g_chunk  # bfloat16
        
        # ============================================================
        # Compute O = Q @ S^T * exp(g) + mask(Q @ K^T * g_attn_matrix) @ delta
        # ============================================================
        
        # o1 = Q @ S^T * exp(g_raw)
        acc_qs = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o1 = ct.mma(q, s_t, acc_qs)
        o1 = o1 * g_out  # Apply output gate
        
        # o2 = mask(Q @ K^T * g_attn_matrix) @ delta
        k_t = ct.transpose(k)
        acc_qk = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, acc_qk)
        qk = qk * g_attn_matrix  # Apply attention gate
        qk_masked = qk * mask  # Apply causal mask
        qk_masked_fp16 = ct.astype(qk_masked, K.dtype)
        
        acc_o2 = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o2 = ct.mma(qk_masked_fp16, delta, acc_o2)
        
        o_chunk = o1 + o2
        o_out = ct.astype(o_chunk, O.dtype)
        
        # Store O
        o_out = ct.reshape(o_out, (1, chunk_size, 1, d))
        ct.store(O, index=(b_idx, t_start, h_idx, 0), tile=o_out)
        
        # ============================================================
        # Update S = S * exp(g_chunk_last) + delta^T @ K
        # ============================================================
        
        delta_t = ct.transpose(delta)
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(delta_t, k, acc_su)
        
        s = s * g_chunk_last_exp + s_update
    
    # Store final state
    s_out = ct.reshape(s, (1, 1, d, d))
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def fused_gdn(
    S: torch.Tensor,    # (B, H, d, d), float32
    T: torch.Tensor,    # (B, seq_len, H, chunk_size), bfloat16
    G: torch.Tensor,    # (B, seq_len, H), bfloat16 - gate values
    Beta: torch.Tensor, # (B, seq_len, H), bfloat16 - beta values
    V: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    Q: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    K: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Fully fused Gated Delta Net forward pass.
    Combines scale_vk and gdn into a single kernel.
    
    Args:
        S: State matrices (B, H, d, d), float32, initialized to zeros
        T: Precomputed T (B, seq_len, H, chunk_size), bfloat16
        G: Gate values (B, seq_len, H), bfloat16
        Beta: Beta values (B, seq_len, H), bfloat16
        V: Values (B, seq_len, H, d), bfloat16
        Q: Queries (B, seq_len, H, d), bfloat16
        K: Keys (B, seq_len, H, d), bfloat16
        chunk_size: Processing chunk size (default 64)
    
    Returns:
        O: Output tensor (B, seq_len, H, d), bfloat16
        (S is updated in-place)
    """
    B_dim, seq_len, H, d = K.shape
    device = K.device
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    assert S.shape == (B_dim, H, d, d), f"S shape mismatch"
    assert S.dtype == torch.float32, "S must be float32"
    assert T.shape == (B_dim, seq_len, H, chunk_size), f"T shape mismatch"
    assert G.shape == (B_dim, seq_len, H), f"G shape mismatch"
    assert G.dtype == torch.bfloat16, "G must be bfloat16"
    assert Beta.shape == (B_dim, seq_len, H), f"Beta shape mismatch"
    assert Beta.dtype == torch.bfloat16, "Beta must be bfloat16"
    
    O = torch.empty(B_dim, seq_len, H, d, dtype=K.dtype, device=device)
    grid = (B_dim * H,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            fused_gdn_kernel,
            (S, T, V, Q, K, G, Beta, O,
             d, chunk_size, num_chunks, H, seq_len)
        )
    
    return O


# Import the separate kernels from gdn.py for comparison
from gdn import scale_vk, gdn, CHUNK_SIZE as GDN_CHUNK_SIZE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()
    
    print("--- Fused GDN Benchmark ---")
    
    B_dim = args.batch
    H = args.heads
    seq_len = args.seq_len
    d = HEAD_DIM
    chunk_size = CHUNK_SIZE
    device = 'cuda'
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Initialize inputs
    S = torch.zeros(B_dim, H, d, d, dtype=torch.float32, device=device)
    V = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    Q = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    K = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    G = torch.sigmoid(torch.randn(B_dim, seq_len, H, dtype=torch.bfloat16, device=device))
    Beta = torch.sigmoid(torch.rand(B_dim, seq_len, H, dtype=torch.bfloat16, device=device))

    # Create dummy T tensor (identity-like for testing)
    T = torch.zeros(B_dim, seq_len, H, chunk_size, dtype=torch.bfloat16, device=device)
    for c in range(num_chunks):
        for i in range(chunk_size):
            t_idx = c * chunk_size + i
            if t_idx < seq_len:
                T[:, t_idx, :, i] = 1.0
    
    print(f"B={B_dim}, H={H}, seq={seq_len}, d={d}, chunks={num_chunks}")
    
    # ============================================================
    # Benchmark 1: Two-Stage (scale_vk + gdn)
    # ============================================================
    print("\n[Two-Stage: scale_vk + gdn]")
    
    # Warmup
    for _ in range(3):
        V_scaled, K_scaled = scale_vk(V, K, G, Beta, tile_size=chunk_size)
        O1 = gdn(torch.zeros_like(S), T, G, V_scaled, Q, K, K_scaled, chunk_size)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Timing - Stage 1: scale_vk
    start.record()
    for _ in range(args.num_iters):
        V_scaled, K_scaled = scale_vk(V, K, G, Beta, tile_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_scale = start.elapsed_time(end) / args.num_iters
    print(f"  scale_vk Time: {ms_scale:.3f} ms")
    
    # Timing - Stage 2: gdn
    start.record()
    for _ in range(args.num_iters):
        O1 = gdn(torch.zeros_like(S), T, G, V_scaled, Q, K, K_scaled, chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_gdn = start.elapsed_time(end) / args.num_iters
    print(f"  gdn Time: {ms_gdn:.3f} ms")
    print(f"  Total: {ms_scale + ms_gdn:.3f} ms")
    
    # ============================================================
    # Benchmark 2: Fully Fused (single kernel)
    # ============================================================
    print("\n[Fully Fused: single kernel]")
    
    # Warmup
    for _ in range(3):
        O2 = fused_gdn(torch.zeros_like(S), T, G, Beta, V, Q, K, chunk_size)
    torch.cuda.synchronize()
    
    # Timing
    start.record()
    for _ in range(args.num_iters):
        O2 = fused_gdn(torch.zeros_like(S), T, G, Beta, V, Q, K, chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_fused = start.elapsed_time(end) / args.num_iters
    print(f"  Fused Time: {ms_fused:.3f} ms")
    
    # ============================================================
    # Comparison
    # ============================================================
    print(f"\nSpeedup: {(ms_scale + ms_gdn) / ms_fused:.2f}x")
    
    # Verify correctness
    diff = (O1 - O2).abs().max().item()
    print(f"Max diff (Two-Stage vs Fused): {diff:.6e}")

