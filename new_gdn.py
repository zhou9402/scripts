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
  S: (batch, num_heads, d, d) float32 - state matrices (initialized to 0)
  T: (batch, seq_len, num_heads, chunk_size) bfloat16 - precomputed T
  G: (batch, seq_len, num_heads) bfloat16 - gate values
  V_scaled: (batch, seq_len, num_heads, d) bfloat16 - scaled values (V * Beta)
  Q: (batch, seq_len, num_heads, d) bfloat16 - queries
  K: (batch, seq_len, num_heads, d) bfloat16 - keys
  K_scaled: (batch, seq_len, num_heads, d) bfloat16 - scaled keys (K * Beta * exp(G))
  
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
def scale_vk_kernel(
    V,        # (B, T, H, d) bfloat16 - input values
    K,        # (B, T, H, d) bfloat16 - input keys
    G,        # (B, T, H) bfloat16 - gate values
    Beta,     # (B, T, H) bfloat16 - beta values
    V_scaled, # (B, T, H, d) bfloat16 - output scaled values
    K_scaled, # (B, T, H, d) bfloat16 - output scaled keys
    d: ConstInt,
    num_heads: ConstInt,
    tile_size: ConstInt,
    num_tiles: ConstInt
):
    """
    Scaling kernel: compute V_scaled = V * Beta, K_scaled = K * Beta * exp(G)
    
    Grid: (B * num_tiles * H,) - one block per tile
    Each block processes one tile of size (tile_size, d).
    """
    linear_idx = ct.bid(0)
    # Decompose: linear_idx = b * (num_tiles * H) + tile_idx * H + h
    b_idx = linear_idx // (num_tiles * num_heads)
    remainder = linear_idx % (num_tiles * num_heads)
    tile_idx = remainder // num_heads
    h_idx = remainder % num_heads
    
    t_start = tile_idx * tile_size
    zero_pad = ct.PaddingMode.ZERO
    
    # Load V: (1, tile_size, 1, d) -> (tile_size, d)
    v = ct.load(V, index=(b_idx, t_start, h_idx, 0), shape=(1, tile_size, 1, d),
                padding_mode=zero_pad)
    v = ct.reshape(v, (tile_size, d))
    
    # Load K: (1, tile_size, 1, d) -> (tile_size, d)
    k = ct.load(K, index=(b_idx, t_start, h_idx, 0), shape=(1, tile_size, 1, d),
                padding_mode=zero_pad)
    k = ct.reshape(k, (tile_size, d))
    
    # Load G: (1, tile_size, 1) -> (tile_size, 1) for broadcast
    g = ct.load(G, index=(b_idx, t_start, h_idx), shape=(1, tile_size, 1),
                padding_mode=zero_pad)
    g = ct.reshape(g, (tile_size, 1))
    
    # Load Beta: (1, tile_size, 1) -> (tile_size, 1) for broadcast
    beta = ct.load(Beta, index=(b_idx, t_start, h_idx), shape=(1, tile_size, 1),
                   padding_mode=zero_pad)
    beta = ct.reshape(beta, (tile_size, 1))
    
    # Compute exp(g) in bfloat16
    exp_g = ct.exp(g)  # bfloat16
    
    # v_scaled = v * beta
    v_s = v * beta  # bfloat16 * bfloat16
    
    # k_scaled = k * beta * exp(g)
    k_s = k * (beta * exp_g)  # bfloat16 * bfloat16 * bfloat16
    
    # Store V_scaled
    v_s = ct.reshape(v_s, (1, tile_size, 1, d))
    ct.store(V_scaled, index=(b_idx, t_start, h_idx, 0), tile=v_s)
    
    # Store K_scaled
    k_s = ct.reshape(k_s, (1, tile_size, 1, d))
    ct.store(K_scaled, index=(b_idx, t_start, h_idx, 0), tile=k_s)


def scale_vk(
    V: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    K: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    G: torch.Tensor,    # (B, seq_len, H), bfloat16
    Beta: torch.Tensor, # (B, seq_len, H), bfloat16
    tile_size: int = CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale V and K by Beta and G (parallel kernel - one block per tile).
    
    Returns:
        V_scaled: V * Beta
        K_scaled: K * Beta * exp(G)
    """
    B_dim, seq_len, H, d = V.shape
    device = V.device
    
    V_scaled = torch.empty_like(V)
    K_scaled = torch.empty_like(K)
    
    num_tiles = (seq_len + tile_size - 1) // tile_size
    # Grid: B * num_tiles * H - one block per tile
    grid = (B_dim * num_tiles * H,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            scale_vk_kernel,
            (V, K, G, Beta, V_scaled, K_scaled,
             d, H, tile_size, num_tiles)
        )
    
    return V_scaled, K_scaled


@ct.kernel
def chunk_kkt_kernel(
    K,        # (B, T, H, d) bfloat16 - keys
    Beta,     # (B, T, H) bfloat16 - beta values
    G,        # (B, T, H) bfloat16 - gate cumsum values
    A,        # (B, T, H, tile_size) bfloat16 - output
    d: ConstInt,
    num_heads: ConstInt,
    tile_size: ConstInt,
    num_tiles: ConstInt,
    total_tiles: ConstInt,
    grid_size: ConstInt
):
    """
    Persistent kernel: Compute A = (K @ K^T) * exp(g[:, None] - g[None, :]) * beta[:, None]
    with strict lower triangular mask (row > col).
    
    Grid: (grid_size,) - fixed number of blocks
    Each block processes multiple tiles using grid-stride loop.
    Output: A has shape (B, T, H, tile_size)
    """
    zero_pad = ct.PaddingMode.ZERO
    
    # Precompute mask outside loop (same for all tiles)
    offs_row = ct.arange(tile_size, dtype=ct.int32)[:, None]  # [tile_size, 1]
    offs_col = ct.arange(tile_size, dtype=ct.int32)[None, :]  # [1, tile_size]
    mask = ct.where(offs_row > offs_col, 1.0, 0.0)  # strict lower triangular
    
    # Grid-stride loop
    for linear_idx in range(ct.bid(0), total_tiles, grid_size):
        # Decompose: linear_idx = b * (num_tiles * H) + tile_idx * H + h
        b_idx = linear_idx // (num_tiles * num_heads)
        remainder = linear_idx % (num_tiles * num_heads)
        tile_idx = remainder // num_heads
        h_idx = remainder % num_heads
        
        t_start = tile_idx * tile_size
        
        # Load K: (tile_size, d)
        k = ct.load(K, index=(b_idx, t_start, h_idx, 0), shape=(1, tile_size, 1, d),
                    padding_mode=zero_pad)
        k = ct.reshape(k, (tile_size, d))
        
        # Load Beta: (tile_size, 1) for broadcast
        beta = ct.load(Beta, index=(b_idx, t_start, h_idx), shape=(1, tile_size, 1),
                       padding_mode=zero_pad)
        beta = ct.reshape(beta, (tile_size, 1))
        
        # Load G: (tile_size, 1) for broadcast
        g = ct.load(G, index=(b_idx, t_start, h_idx), shape=(1, tile_size, 1),
                    padding_mode=zero_pad)
        g = ct.reshape(g, (tile_size, 1))
        
        # Compute K @ K^T
        k_t = ct.transpose(k)  # (d, tile_size)
        acc = ct.full((tile_size, tile_size), 0.0, dtype=ct.float32)
        a = ct.mma(k, k_t, acc)  # (tile_size, tile_size) float32
        a = ct.astype(a, K.dtype)  # convert to bfloat16
        
        # Apply gate: a *= exp(g[:, None] - g[None, :]) - all in bfloat16
        g_t = ct.transpose(g)  # (1, tile_size)
        g_diff = g - g_t  # (tile_size, tile_size) bfloat16
        g_exp = ct.exp(g_diff)  # bfloat16
        a = a * g_exp
        
        # Apply beta: a *= beta[:, None] - bfloat16
        a = a * beta
        
        # Apply strict lower triangular mask (row > col, not >=)
        mask_fp16 = ct.astype(mask, K.dtype)
        a = a * mask_fp16
        
        # Store A: (B, T, H, tile_size)
        a_out = ct.reshape(a, (1, tile_size, 1, tile_size))
        ct.store(A, index=(b_idx, t_start, h_idx, 0), tile=a_out)


def chunk_kkt(
    K: torch.Tensor,    # (B, seq_len, H, d), bfloat16
    Beta: torch.Tensor, # (B, seq_len, H), bfloat16
    G: torch.Tensor,    # (B, seq_len, H), bfloat16 - gate cumsum
    tile_size: int = CHUNK_SIZE,
    num_sms: int = 132  # H100=132, A100=108
) -> torch.Tensor:
    """
    Compute A = (K @ K^T) * exp(g[:, None] - g[None, :]) * beta[:, None]
    with strict lower triangular mask (persistent kernel).
    
    Args:
        K: Keys (B, seq_len, H, d), bfloat16
        Beta: Beta values (B, seq_len, H), bfloat16
        G: Gate cumsum values (B, seq_len, H), bfloat16
        tile_size: Chunk/tile size (default 64)
        num_sms: Number of SMs for persistent kernel (default 132 for H100)
    
    Returns:
        A: Output tensor (B, seq_len, H, tile_size), bfloat16
    """
    B_dim, seq_len, H, d = K.shape
    device = K.device
    
    A = torch.empty(B_dim, seq_len, H, tile_size, dtype=K.dtype, device=device)
    
    num_tiles = (seq_len + tile_size - 1) // tile_size
    total_tiles = B_dim * num_tiles * H
    # Fixed grid size for persistent kernel
    grid_size = min(num_sms, total_tiles)
    grid = (grid_size,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            chunk_kkt_kernel,
            (K, Beta, G, A, d, H, tile_size, num_tiles, total_tiles, grid_size)
        )
    
    return A


@ct.kernel
def gdn_kernel(
    S,        # (B, H, d, d) float32 - state
    T_in,     # (B, T, H, chunk_size) bfloat16 - one vector per timestep
    V_scaled, # (B, T, H, d) bfloat16 - pre-scaled V (V * Beta)
    Q,        # (B, T, H, d) bfloat16 - strided access
    K,        # (B, T, H, d) bfloat16 - original K (for S update and attention)
    K_scaled, # (B, T, H, d) bfloat16 - scaled K (K * Beta * exp(G))
    G,        # (B, T, H) bfloat16 - gate values (still needed for gating)
    O,        # (B, T, H, d) bfloat16 - strided store
    d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    num_chunks: ConstInt,  # number of chunks to process
    num_heads: ConstInt,   # H
    seq_len: ConstInt      # T
):
    """
    GDN kernel: reads pre-scaled V_scaled and K_scaled, plus original K.
    
    Input Layout: V_scaled, Q, K, K_scaled, O: (B, T, H, D) - strided load/store
                  G: (B, T, H) bfloat16 - one scalar per timestep
                  T_in: (B, seq_len, H, chunk_size) - one vector per timestep
    Grid: (B * H,) - one block per (b, h) pair.
    """
    linear_idx = ct.bid(0)
    b_idx = linear_idx // num_heads
    h_idx = linear_idx % num_heads
    zero_pad = ct.PaddingMode.ZERO
    
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.reshape(s, (d, d))
    
    # Create causal mask (lower triangular) inside kernel
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]  # [chunk_size, 1]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]  # [1, chunk_size]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)  # [chunk_size, chunk_size] float32
    
    for c in range(num_chunks):
        t_start = c * chunk_size
        
        # Compute actual chunk size (handle last chunk if not divisible)
        t_end = min(t_start + chunk_size, seq_len)
        
        # T: (B, seq_len, H, chunk_size) -> load chunk as (chunk_size, chunk_size)
        T = ct.load(T_in, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, chunk_size), 
                    padding_mode=zero_pad)
        T = ct.reshape(T, (chunk_size, chunk_size))  # [chunk_size, chunk_size]
        
        # K (original): strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        k = ct.load(K, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        k = ct.reshape(k, (chunk_size, d))
        
        # K_scaled: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        k_scaled = ct.load(K_scaled, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                           padding_mode=zero_pad)
        k_scaled = ct.reshape(k_scaled, (chunk_size, d))
        
        # G: (B, T, H) -> load chunk as (1, chunk_size, 1)
        g_raw = ct.load(G, index=(b_idx, t_start, h_idx), shape=(1, chunk_size, 1), 
                        padding_mode=zero_pad)
        g_raw = ct.reshape(g_raw, (chunk_size, 1))  # [chunk_size, 1] for broadcast
        
        # Get last value in chunk: g_chunk_last = g_raw[actual_chunk - 1]
        g_chunk_last = ct.load(G, index=(b_idx, t_end - 1, h_idx), shape=(1, 1, 1), 
                               padding_mode=zero_pad)
        g_chunk_last = ct.reshape(g_chunk_last, (1, 1))  # scalar as (1, 1)
        
        # Compute gating for delta: g_chunk = exp(g_chunk_last - g_raw)
        g_chunk = ct.exp(g_chunk_last - g_raw)  # [chunk_size, 1]
        # Compute gating for state: g_chunk_last_exp = exp(g_chunk_last)
        g_chunk_last_exp = ct.exp(g_chunk_last)  # [1, 1] scalar
        
        # Compute attention gate matrix: exp(g_raw[:, None] - g_raw[None, :])
        # g_raw is (chunk_size, 1), transpose to get (1, chunk_size)
        g_raw_t = ct.transpose(g_raw)  # [1, chunk_size]
        # Broadcast: (chunk_size, 1) - (1, chunk_size) -> (chunk_size, chunk_size)
        g_attn_matrix = ct.exp(g_raw - g_raw_t)  # [chunk_size, chunk_size]
        
        # Compute output gate: exp(g_raw) for scaling O
        g_out = ct.exp(g_raw)  # [chunk_size, 1]
        
        # V_scaled: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        v_scaled = ct.load(V_scaled, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                           padding_mode=zero_pad)
        v_scaled = ct.reshape(v_scaled, (chunk_size, d))
        
        # Q: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        q = ct.load(Q, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        q = ct.reshape(q, (chunk_size, d))
        
        # W = T @ K_scaled (using pre-scaled K)
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w = ct.astype(ct.mma(T, k_scaled, acc), K.dtype)
        
        # U = T @ V_scaled (using pre-scaled V)
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        u = ct.astype(ct.mma(T, v_scaled, acc), K.dtype)
        
        # delta = U - W @ S^T (all in bfloat16)
        s_fp16 = ct.astype(s, K.dtype)
        s_t = ct.transpose(s_fp16)
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.astype(ct.mma(w, s_t, acc_ws), K.dtype)  # convert to bfloat16
        delta = u - w_st  # bfloat16
        
        # delta = delta * g_chunk, delta is (chunk_size, d), g_chunk is (chunk_size, 1) -> broadcasts
        delta = delta * g_chunk  # bfloat16
        
        # O = Q @ S^T * exp(g_raw) + mask(Q @ K^T * g_attn_matrix) @ delta
        acc_qs = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o1 = ct.mma(q, s_t, acc_qs)
        # Apply output gate only to o1: o1 *= exp(g_raw)
        # g_out is already float32 from exp()
        o1 = o1 * g_out
        
        k_t = ct.transpose(k)
        acc_qk = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, acc_qk)
        # Apply attention gate matrix: qk *= exp(g_raw[:, None] - g_raw[None, :])
        qk = qk * g_attn_matrix
        qk_masked = qk * mask
        qk_masked_fp16 = ct.astype(qk_masked, K.dtype)
        acc_o2 = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o2 = ct.mma(qk_masked_fp16, delta, acc_o2)
        o_chunk = o1 + o2
        o_out = ct.astype(o_chunk, O.dtype)
        
        # O: strided store (B, T, H, d) -> store (1, chunk_size, 1, d)
        o_out = ct.reshape(o_out, (1, chunk_size, 1, d))
        ct.store(O, index=(b_idx, t_start, h_idx, 0), tile=o_out)
        
        # S = S * exp(g_chunk_last) + delta^T @ K (using original K)
        delta_t = ct.transpose(delta)
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(delta_t, k, acc_su)
        
        # g_chunk_last_exp is already float32 from exp(), broadcasts to (d, d)
        s = s * g_chunk_last_exp + s_update
    
    s_out = ct.reshape(s, (1, 1, d, d))
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def gdn(
    S: torch.Tensor,        # (B, H, d, d), float32
    T: torch.Tensor,        # (B, seq_len, H, chunk_size), bfloat16
    G: torch.Tensor,        # (B, seq_len, H), bfloat16 - gate values
    V_scaled: torch.Tensor, # (B, seq_len, H, d), bfloat16 - pre-scaled V
    Q: torch.Tensor,        # (B, seq_len, H, d), bfloat16
    K: torch.Tensor,        # (B, seq_len, H, d), bfloat16 - original K
    K_scaled: torch.Tensor, # (B, seq_len, H, d), bfloat16 - scaled K
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Gated Delta Net forward pass.
    
    Args:
        S: State matrices (B, H, d, d), float32, initialized to zeros
        T: Precomputed T (B, seq_len, H, chunk_size), bfloat16
        G: Gate values (B, seq_len, H), bfloat16
        V_scaled: Pre-scaled values V * Beta (B, seq_len, H, d), bfloat16
        Q: Queries (B, seq_len, H, d), bfloat16
        K: Original keys (B, seq_len, H, d), bfloat16
        K_scaled: Scaled keys K * Beta * exp(G) (B, seq_len, H, d), bfloat16
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
    assert V_scaled.shape == (B_dim, seq_len, H, d), "V_scaled shape mismatch"
    assert K_scaled.shape == (B_dim, seq_len, H, d), "K_scaled shape mismatch"
    
    O = torch.empty(B_dim, seq_len, H, d, dtype=K.dtype, device=device)
    grid = (B_dim * H,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            gdn_kernel,
            (S, T, V_scaled, Q, K, K_scaled, G, O,
             d, chunk_size, num_chunks, H, seq_len)
        )
    
    return O


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
    
    # Pre-scale V and K using separate kernel
    print("\n--- Scale V/K Kernel ---")
    # Warmup
    for _ in range(3):
        V_scaled, K_scaled = scale_vk(V, K, G, Beta, tile_size=chunk_size)
    torch.cuda.synchronize()
    
    # Timing scale_vk
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.num_iters):
        V_scaled, K_scaled = scale_vk(V, K, G, Beta, tile_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_scale = start.elapsed_time(end) / args.num_iters
    print(f"Scale V/K Time: {ms_scale:.3f} ms")
    
    # Chunk K @ K^T kernel
    print("\n--- Chunk K @ K^T Kernel ---")
    # Warmup
    for _ in range(3):
        A = chunk_kkt(K, Beta, G, tile_size=chunk_size)
    torch.cuda.synchronize()
    
    # Timing chunk_kkt
    start.record()
    for _ in range(args.num_iters):
        A = chunk_kkt(K, Beta, G, tile_size=chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_kkt = start.elapsed_time(end) / args.num_iters
    print(f"Chunk K @ K^T Time: {ms_kkt:.3f} ms")
    print(f"A shape: {A.shape}")
    
    # GDN kernel (using pre-scaled inputs)
    print("\n--- GDN Kernel ---")
    # Warmup
    for _ in range(3):
        O = gdn(torch.zeros_like(S), T, G, V_scaled, Q, K, K_scaled, chunk_size)
    torch.cuda.synchronize()
    
    # Timing gdn
    start.record()
    for _ in range(args.num_iters):
        O = gdn(torch.zeros_like(S), T, G, V_scaled, Q, K, K_scaled, chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    ms_gdn = start.elapsed_time(end) / args.num_iters
    print(f"GDN Time: {ms_gdn:.3f} ms")
    
    # Total time
    print("\n--- Total ---")
    print(f"Total Time: {ms_scale + ms_kkt + ms_gdn:.3f} ms (Scale: {ms_scale:.3f} + KKT: {ms_kkt:.3f} + GDN: {ms_gdn:.3f})")
