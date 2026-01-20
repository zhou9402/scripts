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
  S: (batch, num_heads, d, d) bfloat16 - state matrices (initialized to 0)
  T: (batch, seq_len, num_heads, chunk_size) bfloat16 - precomputed T
  G: (batch, seq_len, num_heads) float32 - gate cumsum values
  Beta: (batch, seq_len, num_heads) bfloat16 - beta values
  V: (batch, seq_len, num_heads, d) bfloat16 - values
  Q: (batch, seq_len, num_heads, d) bfloat16 - queries
  K: (batch, seq_len, num_heads, d) bfloat16 - keys
  
Outputs:
  O: (batch, seq_len, num_heads, d) bfloat16 - output
  S: (batch, num_heads, d, d) bfloat16 - updated states (in-place)
"""

import argparse
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# Default configuration
HEAD_DIM = 128    # d
CHUNK_SIZE = 64   # chunk size for processing


def safe_exp(x):
    """Safe exponential: exp(x) if x <= 0, else 0. Prevents overflow."""
    return ct.where(x <= 0.0, ct.exp(x), 0.0)


@ct.kernel
def fused_gdn_kernel(
    S,      # (B, H, d, d) bfloat16 - state
    T_in,   # (B, num_chunks, chunk_size, H, chunk_size) bfloat16 - precomputed T matrix
    V,      # (B, num_chunks, chunk_size, H, d) bfloat16 - values (not pre-scaled)
    Q,      # (B, num_chunks, chunk_size, H, d) bfloat16 - queries
    K,      # (B, num_chunks, chunk_size, H, d) bfloat16 - keys (not pre-scaled)
    G,      # (B, num_chunks, chunk_size, H) float32 - gate cumsum values
    Beta,   # (B, num_chunks, chunk_size, H) bfloat16 - beta values
    O,      # (B, num_chunks, chunk_size, H, d) bfloat16 - output
        d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    H: ConstInt            # num_heads
):
    """
    Fully fused GDN kernel: combines scale_vk and gdn into single kernel.
    
    Mathematical transformation:
      Instead of: W = T @ (K * beta * exp(g)), U = T @ (V * beta)
      We compute: W = (T * beta^T * exp(g)^T) @ K, U = (T * beta^T) @ V
      This saves memory by not storing V_scaled and K_scaled.
    
    Input Layout: All T-dimension tensors reshaped to (B, num_chunks, chunk_size, ...)
    Grid: (B * H,) - one block per (b, h) pair.
    """
    num_chunks = T_in.shape[1]  # Can be dynamic, only used in for loop
    
    linear_idx = ct.bid(0)
    b_idx = linear_idx // H
    h_idx = linear_idx % H
    zero_pad = ct.PaddingMode.ZERO
    
    # Load initial state S and convert to float32 for element-wise operations
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.astype(ct.reshape(s, (d, d)), ct.float32)  # Keep s in float32
    
    # Create causal mask (lower triangular) inside kernel
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]  # [chunk_size, 1]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]  # [1, chunk_size]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)  # [chunk_size, chunk_size] float32
    
    for c in range(num_chunks):
        # ============================================================
        # Load raw inputs for this chunk
        # ============================================================
        
        # T: (B, num_chunks, chunk_size, H, chunk_size) -> (chunk_size, chunk_size)
        T = ct.load(T_in, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, chunk_size), 
                    padding_mode=zero_pad)
        T = ct.reshape(T, (chunk_size, chunk_size))
        
        # K: (B, num_chunks, chunk_size, H, d) -> (chunk_size, d)
        k = ct.load(K, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        k = ct.reshape(k, (chunk_size, d))
        
        # V: (B, num_chunks, chunk_size, H, d) -> (chunk_size, d)
        v = ct.load(V, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        v = ct.reshape(v, (chunk_size, d))
        
        # Q: (B, num_chunks, chunk_size, H, d) -> (chunk_size, d)
        q = ct.load(Q, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        q = ct.reshape(q, (chunk_size, d))
        
        # G: (B, num_chunks, chunk_size, H) -> (chunk_size, 1)
        g_raw = ct.load(G, index=(b_idx, c, 0, h_idx), shape=(1, 1, chunk_size, 1), 
                        padding_mode=zero_pad)
        g_raw = ct.reshape(g_raw, (chunk_size, 1))
        
        # Beta: (B, num_chunks, chunk_size, H) -> (1, chunk_size) for column-wise scaling of T
        beta = ct.load(Beta, index=(b_idx, c, 0, h_idx), shape=(1, 1, chunk_size, 1), 
                       padding_mode=zero_pad)
        beta = ct.reshape(beta, (1, chunk_size))
        
        # Get last g value in chunk for state gating
        g_chunk_last = ct.load(G, index=(b_idx, c, chunk_size - 1, h_idx), shape=(1, 1, 1, 1), 
                               padding_mode=zero_pad)
        g_chunk_last = ct.reshape(g_chunk_last, (1, 1))
        
        # ============================================================
        # Compute gating factors (G is float32 from chunk_local_cumsum)
        # ============================================================
        
        # exp(g) for T_k scaling - bfloat16
        exp_g = ct.exp(g_raw)  # [chunk_size, 1] float32
        
        # g_chunk = safe_exp(g_chunk_last - g_raw) for delta scaling
        g_chunk = safe_exp(g_chunk_last - g_raw)  # [chunk_size, 1] float32
        
        # g_chunk_last_exp = exp(g_chunk_last) for state decay
        g_chunk_last_exp = ct.exp(g_chunk_last)  # [1, 1] scalar float32
        
        # g_attn_matrix = safe_exp(g_raw[:, None] - g_raw[None, :]) for attention
        g_raw_t = ct.transpose(g_raw)  # [1, chunk_size]
        g_attn_matrix = safe_exp(g_raw - g_raw_t)  # [chunk_size, chunk_size] float32
        
        # g_out = exp(g_raw) for output scaling
        g_out = ct.exp(g_raw)  # [chunk_size, 1] float32
        
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
        # exp_g_t is float32, so result is float32, convert back to bfloat16
        T_k = ct.astype(T_v * exp_g_t, K.dtype)  # [chunk_size, chunk_size] bfloat16
        
        # ============================================================
        # Compute W = T_k @ K, U = T_v @ V (using original K, V)
        # ============================================================
        
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w = ct.astype(ct.mma(T_k, k, acc), K.dtype)
        
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        u = ct.astype(ct.mma(T_v, v, acc), K.dtype)
        
        # ============================================================
        # Compute delta = (U - W @ S^T) * g_chunk
        # s is float32, convert to bfloat16 only for mma inputs
        # ============================================================
        
        s_bf16 = ct.astype(s, K.dtype)  # Convert to bfloat16 for mma
        s_t = ct.transpose(s_bf16)
        
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.mma(w, s_t, acc_ws)  # float32 accumulator
        new_v = u - w_st  # float32
        new_v_bf16 = ct.astype(new_v, K.dtype)  # bfloat16 for O calculation
        
        # delta = new_v * safe_exp(g_last - g)
        delta = new_v * g_chunk  # float32
        delta_bf16 = ct.astype(delta, K.dtype)  # bfloat16 for state update
        
        # ============================================================
        # Compute O = Q @ S^T * exp(g) + mask(Q @ K^T * g_attn_matrix) @ new_v
        # Note: FLA's chunk_fwd_o uses v_new (before gate), NOT delta (after gate)
        # ============================================================
        
        # o1 = Q @ S^T * exp(g_raw)
        acc_qs = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o1 = ct.mma(q, s_t, acc_qs) * g_out  # float32
        
        # o2 = mask(Q @ K^T * g_attn_matrix) @ new_v
        k_t = ct.transpose(k)
        acc_qk = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, acc_qk)
        qk_masked = qk * g_attn_matrix * mask  # Apply gate and causal mask
        
        acc_o2 = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o2 = ct.mma(ct.astype(qk_masked, K.dtype), new_v_bf16, acc_o2)
        
        o_out = ct.astype(o1 + o2, O.dtype)
        ct.store(O, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(o_out, (1, 1, chunk_size, 1, d)))
        
        # ============================================================
        # Update S = S * exp(g_chunk_last) + delta^T @ K
        # s is float32, stays float32 for element-wise operations
        # ============================================================
        
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(ct.transpose(delta_bf16), k, acc_su)
        s = s * g_chunk_last_exp + s_update  # fp32 * fp32 + fp32 -> fp32
    
    # Store final state (convert to bfloat16)
    s_out = ct.astype(ct.reshape(s, (1, 1, d, d)), S.dtype)
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def fused_gdn(
    S: torch.Tensor,    # (B, H, d, d), bfloat16
    T: torch.Tensor,    # (B, seq_len, H, chunk_size), bfloat16
    G: torch.Tensor,    # (B, seq_len, H), float32 - gate cumsum values
    Beta: torch.Tensor, # (B, seq_len, H), bfloat16 - beta values
    V: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    Q: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    K: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Fully fused Gated Delta Net forward pass.
    Combines scale_vk and gdn into a single kernel.
    
    Mathematical transformation:
      Instead of: W = T @ (K * beta * exp(g)), U = T @ (V * beta)
      We compute: W = (T * beta^T * exp(g)^T) @ K, U = (T * beta^T) @ V
      This saves memory by not storing V_scaled and K_scaled.
    
    Args:
        S: State matrices (B, H, d, d), bfloat16, initialized to zeros
        T: Precomputed T (B, seq_len, H, chunk_size), bfloat16
        G: Gate cumsum values (B, seq_len, H), float32
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
    num_chunks = seq_len // chunk_size
    
    assert S.shape == (B_dim, H, d, d), f"S shape mismatch"
    assert S.dtype == torch.bfloat16, "S must be bfloat16"
    assert T.shape == (B_dim, seq_len, H, chunk_size), f"T shape mismatch"
    assert G.shape == (B_dim, seq_len, H), f"G shape mismatch"
    assert G.dtype == torch.float32, "G must be float32"
    assert Beta.shape == (B_dim, seq_len, H), f"Beta shape mismatch"
    assert Beta.dtype == torch.bfloat16, "Beta must be bfloat16"
    
    # Reshape inputs to (B, num_chunks, chunk_size, ...)
    T_reshaped = T.reshape(B_dim, num_chunks, chunk_size, H, chunk_size)
    V_reshaped = V.reshape(B_dim, num_chunks, chunk_size, H, d)
    Q_reshaped = Q.reshape(B_dim, num_chunks, chunk_size, H, d)
    K_reshaped = K.reshape(B_dim, num_chunks, chunk_size, H, d)
    G_reshaped = G.reshape(B_dim, num_chunks, chunk_size, H)
    Beta_reshaped = Beta.reshape(B_dim, num_chunks, chunk_size, H)
    
    # Output shape: (B, num_chunks, chunk_size, H, d)
    O_reshaped = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=device)
    
    grid = (B_dim * H,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            fused_gdn_kernel,
            (S, T_reshaped, V_reshaped, Q_reshaped, K_reshaped, G_reshaped, Beta_reshaped, O_reshaped,
             d, chunk_size, H)
        )
    
    # Reshape output back to (B, seq_len, H, d)
    O = O_reshaped.reshape(B_dim, seq_len, H, d)
    
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
    S = torch.zeros(B_dim, H, d, d, dtype=torch.bfloat16, device=device)
    V = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    Q = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    K = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    G = torch.sigmoid(torch.randn(B_dim, seq_len, H, dtype=torch.float32, device=device))  # float32 gate cumsum
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
        V_scaled_r, K_scaled_r = scale_vk(V, K, G, Beta, chunk_size=chunk_size)
        O1 = gdn(torch.zeros_like(S), T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Timing - Stage 1: scale_vk
    start.record()
    for _ in range(args.num_iters):
        V_scaled_r, K_scaled_r = scale_vk(V, K, G, Beta, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_scale = start.elapsed_time(end) / args.num_iters
    print(f"  scale_vk Time: {ms_scale:.3f} ms")
    
    # Timing - Stage 2: gdn
    start.record()
    for _ in range(args.num_iters):
        O1 = gdn(torch.zeros_like(S), T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size)
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
    
    # ============================================================
    # Verification: Check if outputs are close
    # ============================================================
    print("\n--- Verification ---")
    
    # Compute fresh outputs for comparison (not from timing loop)
    S1 = torch.zeros_like(S)
    S2 = torch.zeros_like(S)
    V_scaled_r, K_scaled_r = scale_vk(V, K, G, Beta, chunk_size=chunk_size)
    O1_fresh = gdn(S1, T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size)
    O2_fresh = fused_gdn(S2, T, G, Beta, V, Q, K, chunk_size)
    
    # Output comparison
    diff_abs = (O1_fresh - O2_fresh).abs()
    max_diff = diff_abs.max().item()
    mean_diff = diff_abs.mean().item()
    
    # Relative error (avoid div by zero)
    O1_abs = O1_fresh.abs()
    rel_err = (diff_abs / (O1_abs + 1e-8)).mean().item()
    
    # Check with different tolerances
    rtol, atol = 1e-2, 1e-3  # Relaxed tolerance for bfloat16
    is_close = torch.allclose(O1_fresh, O2_fresh, rtol=rtol, atol=atol)
    
    print(f"Output O:")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")
    print(f"  Mean rel err:  {rel_err:.6e}")
    print(f"  torch.allclose(rtol={rtol}, atol={atol}): {is_close}")
    
    # State comparison
    diff_s = (S1 - S2).abs()
    max_diff_s = diff_s.max().item()
    mean_diff_s = diff_s.mean().item()
    is_close_s = torch.allclose(S1, S2, rtol=rtol, atol=atol)
    
    print(f"State S:")
    print(f"  Max abs diff:  {max_diff_s:.6e}")
    print(f"  Mean abs diff: {mean_diff_s:.6e}")
    print(f"  torch.allclose(rtol={rtol}, atol={atol}): {is_close_s}")
    
    if is_close and is_close_s:
        print("\n✓ PASSED: Fused and Two-Stage outputs match!")
    else:
        print("\n✗ FAILED: Outputs differ beyond tolerance!")

