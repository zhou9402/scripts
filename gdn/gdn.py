# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated Delta Net: Fused chunk_gated_delta_rule_fwd_h and chunk_fwd_o

Layout: (B, T, H, D) - Batch, Time/Sequence, Head, Dimension
        No reshape needed - kernel directly loads strided data

Algorithm:
  Preprocessing (separate kernel):
    V_scaled = V * Beta
    K_scaled = K * Beta * exp(G)

  For each chunk:
    W = T @ K_scaled            # [chunk, d], using pre-scaled K
    U = T @ V_scaled            # [chunk, d], using pre-scaled V
    delta = (U - W @ S^T) * exp(g_last - g)   # [chunk, d], gated delta
    O = (Q @ S^T) * exp(g) + mask(Q @ K^T * exp(g[:, None] - g[None, :])) @ delta
    S = S * exp(g_last) + delta^T @ K   # [d, d], state update with decay

Inputs:
  S: (batch, num_heads, d, d) bfloat16 - state matrices (initialized to 0)
  T: (batch, seq_len, num_heads, chunk_size) bfloat16 - precomputed T
  G: (batch, seq_len, num_heads) float32 - gate cumsum values
  V_scaled: (batch, seq_len, num_heads, d) bfloat16 - scaled values (V * Beta)
  Q: (batch, seq_len, num_heads, d) bfloat16 - queries
  K: (batch, seq_len, num_heads, d) bfloat16 - keys
  K_scaled: (batch, seq_len, num_heads, d) bfloat16 - scaled keys (K * Beta * exp(G))
  
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
    # Avoid ct.full(x.shape) which may not work correctly in cuda.tile
    return ct.where(x <= 0.0, ct.exp(x), 0.0)


@ct.kernel
def scale_vk_kernel(
    V,        # (B, num_chunks, chunk_size, H, d) bfloat16 - input values
    K,        # (B, num_chunks, chunk_size, H, d) bfloat16 - input keys
    G,        # (B, num_chunks, chunk_size, H) float32 - gate values
    Beta,     # (B, num_chunks, chunk_size, H) bfloat16 - beta values
    V_scaled, # (B, num_chunks, chunk_size, H, d) bfloat16 - output scaled values
    K_scaled, # (B, num_chunks, chunk_size, H, d) bfloat16 - output scaled keys
    d: ConstInt, chunk_size: ConstInt
):
    """Compute V_scaled = V * Beta, K_scaled = K * Beta * exp(G)
    
    Parallel kernel: one block per tile.
    Layout: (B, num_chunks, chunk_size, H, d)
    Grid: (B * num_chunks * H,)
    """
    num_chunks = V.shape[1]
    H = V.shape[3]
    zero_pad = ct.PaddingMode.ZERO
    
    idx = ct.bid(0)
    b_idx = idx // (num_chunks * H)
    remainder = idx % (num_chunks * H)
    c_idx = remainder // H
    h_idx = remainder % H
    
    # Load V: (B, num_chunks, chunk_size, H, d) -> (chunk_size, d)
    v = ct.load(V, index=(b_idx, c_idx, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad)
    v = ct.reshape(v, (chunk_size, d))
    
    # Load Beta: (B, num_chunks, chunk_size, H) -> (chunk_size, 1)
    beta = ct.load(Beta, index=(b_idx, c_idx, 0, h_idx), shape=(1, 1, chunk_size, 1), padding_mode=zero_pad)
    beta = ct.reshape(beta, (chunk_size, 1))
    
    v_s = v * beta

    # Load K: (B, num_chunks, chunk_size, H, d) -> (chunk_size, d)
    k = ct.load(K, index=(b_idx, c_idx, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad)
    k = ct.reshape(k, (chunk_size, d))
    
    # Load G: (B, num_chunks, chunk_size, H) -> (chunk_size, 1)
    g = ct.load(G, index=(b_idx, c_idx, 0, h_idx), shape=(1, 1, chunk_size, 1), padding_mode=zero_pad)
    g = ct.reshape(g, (chunk_size, 1))
    exp_g = ct.exp(g)
    k_s = k * beta * exp_g
    
    # Ensure bfloat16 before store
    v_s = ct.astype(v_s, V_scaled.dtype)
    k_s = ct.astype(k_s, K_scaled.dtype)
    
    # Store
    ct.store(V_scaled, index=(b_idx, c_idx, 0, h_idx, 0), tile=ct.reshape(v_s, (1, 1, chunk_size, 1, d)))
    ct.store(K_scaled, index=(b_idx, c_idx, 0, h_idx, 0), tile=ct.reshape(k_s, (1, 1, chunk_size, 1, d)))


def scale_vk(V, K, G, Beta, chunk_size=CHUNK_SIZE):
    """Scale V and K with reshaping to (B, num_chunks, chunk_size, H, d)
    
    Parallel kernel: one block per tile.
    Returns reshaped tensors directly (no reshape back).
    """
    B_dim, seq_len, H, d = V.shape
    num_chunks = seq_len // chunk_size
    total_tiles = B_dim * num_chunks * H
    
    # Reshape to (B, num_chunks, chunk_size, H, d/-)
    V_r = V.reshape(B_dim, num_chunks, chunk_size, H, d)
    K_r = K.reshape(B_dim, num_chunks, chunk_size, H, d)
    G_r = G.reshape(B_dim, num_chunks, chunk_size, H)
    Beta_r = Beta.reshape(B_dim, num_chunks, chunk_size, H)
    
    V_scaled_r = torch.empty_like(V_r)
    K_scaled_r = torch.empty_like(K_r)
    
    with torch.cuda.device(V.device):
        ct.launch(torch.cuda.current_stream(), (total_tiles,), scale_vk_kernel,
                  (V_r, K_r, G_r, Beta_r, V_scaled_r, K_scaled_r, d, chunk_size))
    
    # Return reshaped tensors directly
    return V_scaled_r, K_scaled_r


@ct.kernel
def chunk_kkt_kernel(
    K,        # (B, num_chunks, chunk_size, H, d) bfloat16 - keys
    Beta,     # (B, num_chunks, chunk_size, H) bfloat16 - beta values
    G,        # (B, num_chunks, chunk_size, H) float32 - gate cumsum values
    A,        # (B, num_chunks, chunk_size, H, chunk_size) bfloat16 - output
    d: ConstInt,
    chunk_size: ConstInt
):
    """
    Compute A = (K @ K^T) * exp(g[:, None] - g[None, :]) * beta[:, None]
    with strict lower triangular mask (row > col).
    
    Grid: (B * num_chunks * H,) - one block per chunk
    """
    zero_pad = ct.PaddingMode.ZERO
    
    # Get dimensions from tensor shapes
    H = K.shape[3]
    num_chunks = K.shape[1]
    
    linear_idx = ct.bid(0)
    # Decompose linear index into (b_idx, chunk_idx, h_idx)
    # Layout: linear_idx = b_idx * (num_chunks * H) + chunk_idx * H + h_idx
    b_idx = linear_idx // (num_chunks * H)
    remainder = linear_idx % (num_chunks * H)
    chunk_idx = remainder // H
    h_idx = remainder % H
    
    # Compute mask (independent of tile index, computed once)
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]  # [chunk_size, 1]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]  # [1, chunk_size]
    mask = ct.where(offs_row > offs_col, 1.0, 0.0)  # strict lower triangular
    mask_fp16 = ct.astype(mask, K.dtype)
    
    # Load K: (chunk_size, d) - directly index by chunk_idx
    k = ct.load(K, index=(b_idx, chunk_idx, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d),
                padding_mode=zero_pad)
    k = ct.reshape(k, (chunk_size, d))
    
    # Load Beta: (chunk_size, 1) for broadcast
    beta = ct.load(Beta, index=(b_idx, chunk_idx, 0, h_idx), shape=(1, 1, chunk_size, 1),
                   padding_mode=zero_pad)
    beta = ct.reshape(beta, (chunk_size, 1))
    
    # Load G: (chunk_size, 1) for broadcast
    g = ct.load(G, index=(b_idx, chunk_idx, 0, h_idx), shape=(1, 1, chunk_size, 1),
                padding_mode=zero_pad)
    g = ct.reshape(g, (chunk_size, 1))
    
    # Compute K @ K^T
    k_t = ct.transpose(k)  # (d, chunk_size)
    acc = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
    a = ct.mma(k, k_t, acc)  # (chunk_size, chunk_size) float32
    a = ct.astype(a, K.dtype)  # convert to bfloat16
    
    # Apply gate: a *= exp(g[:, None] - g[None, :]) - all in bfloat16
    g_t = ct.transpose(g)  # (1, chunk_size)
    g_diff = g - g_t  # (chunk_size, chunk_size) float32 (g is fp32)
    g_exp = ct.exp(g_diff)  # float32
    a = a * g_exp  # bf16 * fp32 -> fp32
    
    # Apply beta: a *= beta[:, None] - bfloat16
    a = a * beta # fp32 * bf16 -> fp32
    
    # Apply strict lower triangular mask (row > col, not >=)
    a = a * mask_fp16 # fp32 * bf16 -> fp32
    
    # Store A: (B, num_chunks, chunk_size, H, chunk_size)
    a_out = ct.astype(a, A.dtype)  # Ensure bfloat16
    a_out = ct.reshape(a_out, (1, 1, chunk_size, 1, chunk_size))
    ct.store(A, index=(b_idx, chunk_idx, 0, h_idx, 0), tile=a_out)


def chunk_kkt(
    K: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    Beta: torch.Tensor, # (B, seq_len, H), bfloat16
    G: torch.Tensor,    # (B, seq_len, H), float32 - gate cumsum
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Compute A = (K @ K^T) * exp(g[:, None] - g[None, :]) * beta[:, None]
    with strict lower triangular mask.
    """
    B_dim, seq_len, H, d = K.shape
    device = K.device
    num_chunks = seq_len // chunk_size
    
    # Reshape inputs to (B, num_chunks, chunk_size, ...)
    K_reshaped = K.reshape(B_dim, num_chunks, chunk_size, H, d)
    Beta_reshaped = Beta.reshape(B_dim, num_chunks, chunk_size, H)
    G_reshaped = G.reshape(B_dim, num_chunks, chunk_size, H)
    
    # Output shape: (B, num_chunks, chunk_size, H, chunk_size)
    A_reshaped = torch.empty(B_dim, num_chunks, chunk_size, H, chunk_size, dtype=K.dtype, device=device)
    
    # Normal grid - one block per chunk
    grid_size = B_dim * num_chunks * H
    grid = (grid_size,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            chunk_kkt_kernel,
            (K_reshaped, Beta_reshaped, G_reshaped, A_reshaped, d, chunk_size)
        )
    
    # Reshape output back to (B, seq_len, H, chunk_size)
    A = A_reshaped.reshape(B_dim, seq_len, H, chunk_size)
    
    return A


@ct.kernel
def gdn_kernel(
    S,        # (B, H, d, d) bfloat16 - state
    T_in,     # (B, num_chunks, chunk_size, H, chunk_size) bfloat16
    V_scaled, # (B, num_chunks, chunk_size, H, d) bfloat16 - pre-scaled V (V * Beta)
    Q,        # (B, num_chunks, chunk_size, H, d) bfloat16 - queries
    K,        # (B, num_chunks, chunk_size, H, d) bfloat16 - original K
    K_scaled, # (B, num_chunks, chunk_size, H, d) bfloat16 - scaled K (K * Beta * exp(G))
    G,        # (B, num_chunks, chunk_size, H) float32 - gate values
    O_out,    # (B, num_chunks, chunk_size, H, d) bfloat16 - output
    d: ConstInt, chunk_size: ConstInt
):
    """GDN kernel: computes O and updates S in-place.
    
    O = Q @ S^T * exp(g) + mask(Q @ K^T * exp(g[:, None] - g[None, :])) @ delta
    State S is bfloat16 (matching FLA's h tensor storage).
    Internal accumulation uses float32 for precision.
    """
    H = S.shape[1]
    num_chunks = T_in.shape[1]
    
    idx = ct.bid(0)
    b_idx = idx // H
    h_idx = idx % H
    zero_pad = ct.PaddingMode.ZERO
    
    s = ct.reshape(ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad), (d, d))
    
    # Create causal mask (lower triangular) - row >= col
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)  # [chunk_size, chunk_size] float32
    
    for c in range(num_chunks):
        T = ct.reshape(ct.load(T_in, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, chunk_size), padding_mode=zero_pad), (chunk_size, chunk_size))
        q = ct.reshape(ct.load(Q, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        k = ct.reshape(ct.load(K, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        k_scaled = ct.reshape(ct.load(K_scaled, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        g_raw = ct.reshape(ct.load(G, index=(b_idx, c, 0, h_idx), shape=(1, 1, chunk_size, 1), padding_mode=zero_pad), (chunk_size, 1))
        g_chunk_last = ct.reshape(ct.load(G, index=(b_idx, c, chunk_size - 1, h_idx), shape=(1, 1, 1, 1), padding_mode=zero_pad), (1, 1))
        
        # Gate computations - g is float32 from chunk_local_cumsum, exp returns float32
        g_chunk = safe_exp(g_chunk_last - g_raw)  # fp32, safe_exp for delta
        g_chunk_last_exp = ct.exp(g_chunk_last)  # fp32 for state update
        g_out = ct.exp(g_raw)  # fp32 for O scaling
        g_attn = safe_exp(g_raw - ct.transpose(g_raw))  # [chunk_size, chunk_size] fp32 for attention
        
        v_scaled = ct.reshape(ct.load(V_scaled, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        
        # W = T @ K_scaled, U = T @ V_scaled (both to bfloat16 like FLA's recompute_w_u_fwd output)
        w = ct.astype(ct.mma(T, k_scaled, ct.full((chunk_size, d), 0.0, dtype=ct.float32)), K.dtype)
        u = ct.astype(ct.mma(T, v_scaled, ct.full((chunk_size, d), 0.0, dtype=ct.float32)), K.dtype)
        
        # new_v = U - W @ S^T (s is bfloat16)
        s_t = ct.transpose(s)
        w_st = ct.mma(w, s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32))  # fp32 accumulator
        new_v = u - w_st  # float32
        
        # delta = new_v * safe_exp(g_last - g), for state update and O calculation
        delta = new_v * g_chunk  # float32
        delta_bf16 = ct.astype(delta, K.dtype)  # bfloat16 for mma
        
        # O = Q @ S^T * exp(g) + mask(Q @ K^T * g_attn) @ delta
        o1 = ct.mma(q, s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32)) * g_out
        qk = ct.mma(q, ct.transpose(k), ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32))
        qk_masked = qk * g_attn * mask  # apply gate and causal mask
        o2 = ct.mma(ct.astype(qk_masked, K.dtype), delta_bf16, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        o_out = ct.astype(o1 + o2, O_out.dtype)
        ct.store(O_out, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(o_out, (1, 1, chunk_size, 1, d)))
        
        # S = S * exp(g_last) + delta^T @ K
        s_update = ct.mma(ct.transpose(delta_bf16), k, ct.full((d, d), 0.0, dtype=ct.float32))
        s_fp32 = s * g_chunk_last_exp + s_update  # bf16 * fp32 + fp32 -> fp32
        s = ct.astype(s_fp32, S.dtype)  # Convert back to bfloat16
    
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=ct.reshape(s, (1, 1, d, d)))


def gdn(S, T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size=CHUNK_SIZE):
    """
    GDN forward pass: computes O and updates S in-place.
    
    Args:
        S: State (B, H, d, d), bfloat16
        T: Precomputed T (B, seq_len, H, chunk_size), bfloat16
        G: Gate cumsum (B, seq_len, H), float32
        V_scaled_r: V * Beta, already reshaped to (B, num_chunks, chunk_size, H, d), bfloat16
        Q: Queries (B, seq_len, H, d), bfloat16
        K: Original keys (B, seq_len, H, d), bfloat16
        K_scaled_r: K * Beta * exp(G), already reshaped to (B, num_chunks, chunk_size, H, d), bfloat16
    
    Returns: O (B, seq_len, H, d), bfloat16
    """
    B_dim, seq_len, H, d = K.shape
    num_chunks = seq_len // chunk_size
    
    T_r = T.reshape(B_dim, num_chunks, chunk_size, H, chunk_size)
    Q_r = Q.reshape(B_dim, num_chunks, chunk_size, H, d)
    K_r = K.reshape(B_dim, num_chunks, chunk_size, H, d)
    G_r = G.reshape(B_dim, num_chunks, chunk_size, H)
    O_r = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=K.device)
    
    with torch.cuda.device(K.device):
        ct.launch(torch.cuda.current_stream(), (B_dim * H,), gdn_kernel,
                  (S, T_r, V_scaled_r, Q_r, K_r, K_scaled_r, G_r, O_r, d, chunk_size))
    
    return O_r.reshape(B_dim, seq_len, H, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()
    
    print("--- GDN Benchmark ---")
    
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
    G = torch.sigmoid(torch.randn(B_dim, seq_len, H, dtype=torch.float32, device=device))
    Beta = torch.sigmoid(torch.rand(B_dim, seq_len, H, dtype=torch.bfloat16, device=device))

    # Create dummy T tensor (identity-like for testing)
    T = torch.zeros(B_dim, seq_len, H, chunk_size, dtype=torch.bfloat16, device=device)
    for c in range(num_chunks):
        for i in range(chunk_size):
            t_idx = c * chunk_size + i
            if t_idx < seq_len:
                T[:, t_idx, :, i] = 1.0
    
    print(f"B={B_dim}, H={H}, seq={seq_len}, d={d}, chunks={num_chunks}")
    
    # Pre-scale V and K using separate kernel
    print("\n--- Scale V/K Kernel ---")
    # Warmup
    for _ in range(3):
        V_scaled_r, K_scaled_r = scale_vk(V, K, G, Beta, chunk_size=chunk_size)
    torch.cuda.synchronize()
    
    # Timing scale_vk
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.num_iters):
        V_scaled_r, K_scaled_r = scale_vk(V, K, G, Beta, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_scale = start.elapsed_time(end) / args.num_iters
    print(f"Scale V/K Time: {ms_scale:.3f} ms")
    
    # Chunk K @ K^T kernel
    print("\n--- Chunk K @ K^T Kernel ---")
    # Warmup
    for _ in range(3):
        A = chunk_kkt(K, Beta, G, chunk_size=chunk_size)
    torch.cuda.synchronize()
    
    # Timing chunk_kkt
    start.record()
    for _ in range(args.num_iters):
        A = chunk_kkt(K, Beta, G, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_kkt = start.elapsed_time(end) / args.num_iters
    print(f"Chunk K @ K^T Time: {ms_kkt:.3f} ms")
    print(f"A shape: {A.shape}")
    
    # GDN kernel (using pre-scaled inputs)
    print("\n--- GDN Kernel ---")
    # Warmup
    for _ in range(3):
        O = gdn(torch.zeros_like(S), T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size)
    torch.cuda.synchronize()
    
    # Timing gdn
    start.record()
    for _ in range(args.num_iters):
        O = gdn(torch.zeros_like(S), T, G, V_scaled_r, Q, K, K_scaled_r, chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_gdn = start.elapsed_time(end) / args.num_iters
    print(f"GDN Time: {ms_gdn:.3f} ms")
    
    # Total time
    print("\n--- Total ---")
    print(f"Total Time: {ms_scale + ms_kkt + ms_gdn:.3f} ms (Scale: {ms_scale:.3f} + KKT: {ms_kkt:.3f} + GDN: {ms_gdn:.3f})")
