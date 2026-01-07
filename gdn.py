# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated Delta Net: Fused chunk_gated_delta_rule_fwd_h and chunk_fwd_o

Layout: (B, T, H, D) - Batch, Time/Sequence, Head, Dimension
        No reshape needed - kernel directly loads strided data

Algorithm:
  For each chunk:
    delta = U - W @ S^T            # [chunk, d]
    O     = Q @ S^T + mask(Q @ K^T) @ delta   # [chunk, d], mask is causal (lower triangular)
    S     = S + delta^T @ K        # [d, d], state update

Key optimizations:
1. Pipeline: O uses previous S, so O computation can overlap with next chunk's data load
2. Reuse delta: delta is used in both O and S update, compute once
3. Memory bound: maximize memory requests, use TMA with latency hints
4. Multi-batch: each block handles one batch independently
5. Strided load: directly load (B, T, H, D) without reshape

Inputs:
  S: (batch, num_heads, d, d) float32 - state matrices (initialized to 0)
  W: (batch, seq_len, num_heads, d) bfloat16 - gate weights
  U: (batch, seq_len, num_heads, d) bfloat16 - update values
  Q: (batch, seq_len, num_heads, d) bfloat16 - queries
  K: (batch, seq_len, num_heads, d) bfloat16 - keys
  
Outputs:
  O: (batch, seq_len, num_heads, d) bfloat16 - output
  S: (batch, num_heads, d, d) float32 - updated states (in-place)
"""

import argparse
import cuda.tile as ct
import torch
from math import ceil

ConstInt = ct.Constant[int]

# Default configuration
HEAD_DIM = 128    # d
CHUNK_SIZE = 64   # chunk size for processing

def solve_16x16_block(a_blk, identity_16, dtype):
    """Compute (I + A)^{-1} for 16x16 strictly lower triangular block via Neumann series.
    
    (I + A)^-1 = I - A + A^2 - A^3 + ... (A is already strictly lower triangular)
    Inputs are fp16, accumulation in fp32.
    """
    neg_a = -a_blk  # fp16
    
    # (I + A)^-1 = I + (-A) + (-A)^2 + (-A)^3 + ...
    # Start with I + (-A) in fp32
    result = ct.astype(identity_16, ct.float32) + ct.astype(neg_a, ct.float32)
    
    # p2 = (-A)^2, accumulate in fp32
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    p2 = ct.mma(neg_a, neg_a, acc)
    result = result + p2
    p2_fp16 = ct.astype(p2, dtype)
    
    # p3 = (-A)^3
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    p3 = ct.mma(p2_fp16, neg_a, acc)
    result = result + p3
    p3_fp16 = ct.astype(p3, dtype)
    
    # p4 = (-A)^4
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    p4 = ct.mma(p2_fp16, p2_fp16, acc)
    result = result + p4
    p4_fp16 = ct.astype(p4, dtype)
    
    # p5 = (-A)^5, p6 = (-A)^6, p7 = (-A)^7
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    result = result + ct.mma(p4_fp16, neg_a, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    result = result + ct.mma(p4_fp16, p2_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    result = result + ct.mma(p4_fp16, p3_fp16, acc)
    
    # Convert final result to fp16
    return ct.astype(result, dtype)


def solve_tril_64x64(A, identity_16, dtype):
    """
    Compute T = (I + A)^{-1} for 64x64 strictly lower triangular A.
    Uses 16x16 block decomposition and merge.
    
    Args:
        A: 64x64 strictly lower triangular matrix (fp16)
        identity_16: 16x16 identity matrix (fp16)
        dtype: output dtype (K.dtype)
    
    Returns:
        T: 64x64 inverse matrix (fp16)
    """
    # Extract 16x16 blocks from A
    a11 = ct.extract(A, index=(0, 0), shape=(16, 16))
    a22 = ct.extract(A, index=(16, 16), shape=(16, 16))
    a33 = ct.extract(A, index=(32, 32), shape=(16, 16))
    a44 = ct.extract(A, index=(48, 48), shape=(16, 16))
    a21 = ct.extract(A, index=(16, 0), shape=(16, 16))
    a31 = ct.extract(A, index=(32, 0), shape=(16, 16))
    a32 = ct.extract(A, index=(32, 16), shape=(16, 16))
    a41 = ct.extract(A, index=(48, 0), shape=(16, 16))
    a42 = ct.extract(A, index=(48, 16), shape=(16, 16))
    a43 = ct.extract(A, index=(48, 32), shape=(16, 16))
    
    # Solve diagonal blocks: T_ii = (I + A_ii)^{-1}
    t11 = solve_16x16_block(a11, identity_16, dtype)
    t22 = solve_16x16_block(a22, identity_16, dtype)
    t33 = solve_16x16_block(a33, identity_16, dtype)
    t44 = solve_16x16_block(a44, identity_16, dtype)
    
    t11_fp16 = ct.astype(t11, dtype)
    t22_fp16 = ct.astype(t22, dtype)
    t33_fp16 = ct.astype(t33, dtype)
    t44_fp16 = ct.astype(t44, dtype)
    
    # Off-diagonal blocks: T_ij = -T_ii @ (sum_k A_ik @ T_kj)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a21, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t21_fp16 = ct.astype(-ct.mma(t22_fp16, ct.astype(tmp, dtype), acc), dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a32, t22_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t32_fp16 = ct.astype(-ct.mma(t33_fp16, ct.astype(tmp, dtype), acc), dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a43, t33_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t43_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp, dtype), acc), dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a31, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a32, t21_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t31_fp16 = ct.astype(-ct.mma(t33_fp16, ct.astype(tmp1 + tmp2, dtype), acc), dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a42, t22_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a43, t32_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t42_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp1 + tmp2, dtype), acc), dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a41, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a42, t21_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp3 = ct.mma(a43, t31_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t41_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp1 + tmp2 + tmp3, dtype), acc), dtype)
    
    # Assemble T (64x64) using cat
    zero16 = ct.full((16, 16), 0.0, dtype=dtype)
    row0_a = ct.cat((t11_fp16, zero16), axis=1)
    row0_b = ct.cat((zero16, zero16), axis=1)
    row0 = ct.cat((row0_a, row0_b), axis=1)
    row1_a = ct.cat((t21_fp16, t22_fp16), axis=1)
    row1_b = ct.cat((zero16, zero16), axis=1)
    row1 = ct.cat((row1_a, row1_b), axis=1)
    row2_a = ct.cat((t31_fp16, t32_fp16), axis=1)
    row2_b = ct.cat((t33_fp16, zero16), axis=1)
    row2 = ct.cat((row2_a, row2_b), axis=1)
    row3_a = ct.cat((t41_fp16, t42_fp16), axis=1)
    row3_b = ct.cat((t43_fp16, t44_fp16), axis=1)
    row3 = ct.cat((row3_a, row3_b), axis=1)
    top = ct.cat((row0, row1), axis=0)
    bottom = ct.cat((row2, row3), axis=0)
    T = ct.cat((top, bottom), axis=0)
    
    return T


@ct.kernel
def gated_delta_net_kernel(
    S,      # (B, H, d, d) float32 - state
    T_in,   # (B, T, H, chunk_size) bfloat16 - one vector per timestep
    V,      # (B, T, H, d) bfloat16 - strided access
    Q,      # (B, T, H, d) bfloat16 - strided access
    K,      # (B, T, H, d) bfloat16 - strided access
    G,      # (B, num_chunks, H, d) bfloat16 - gate values, one vector per chunk
    O,      # (B, T, H, d) bfloat16 - strided store
    causal_mask,  # (chunk_size, chunk_size) float32 - precomputed lower triangular
    d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    num_chunks: ConstInt,  # number of chunks to process
    num_heads: ConstInt    # H
):
    """
    Two-stage GDN kernel: T is precomputed and loaded from global memory.
    
    Input Layout: V, Q, K, O: (B, T, H, D) - strided load/store
                  G: (B, num_chunks, H, D) - one gate vector per chunk
                  T_in: (B, seq_len, H, chunk_size) - one vector per timestep
    Grid: (B * H,) - one block per (b, h) pair.
    """
    linear_idx = ct.bid(0)
    b_idx = linear_idx // num_heads
    h_idx = linear_idx % num_heads
    zero_pad = ct.PaddingMode.ZERO
    
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.reshape(s, (d, d))
    mask = ct.load(causal_mask, index=(0, 0), shape=(chunk_size, chunk_size), padding_mode=zero_pad)
    
    for c in range(num_chunks):
        t_start = c * chunk_size
        
        # T: (B, seq_len, H, chunk_size) -> load chunk as (chunk_size, chunk_size)
        T = ct.load(T_in, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, chunk_size), 
                    padding_mode=zero_pad)
        T = ct.reshape(T, (chunk_size, chunk_size))  # [chunk_size, chunk_size]
        
        # K: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        k = ct.load(K, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        k = ct.reshape(k, (chunk_size, d))
        
        # G: (B, num_chunks, H, d) -> load (1, 1, 1, d) - one gate vector per chunk
        gate = ct.load(G, index=(b_idx, c, h_idx, 0), shape=(1, 1, 1, d), 
                       padding_mode=zero_pad)
        gate = ct.reshape(gate, (1, d))  # [1, d] for broadcast across rows
        
        # V: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        v = ct.load(V, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        v = ct.reshape(v, (chunk_size, d))
        
        # Q: strided (B, T, H, d) -> load (1, chunk_size, 1, d)
        q = ct.load(Q, index=(b_idx, t_start, h_idx, 0), shape=(1, chunk_size, 1, d), 
                    padding_mode=zero_pad)
        q = ct.reshape(q, (chunk_size, d))
        
        # W = T @ K, U = T @ V
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w = ct.astype(ct.mma(T, k, acc), K.dtype)
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        u = ct.astype(ct.mma(T, v, acc), K.dtype)
        
        # delta = U - W @ S^T
        s_fp16 = ct.astype(s, K.dtype)
        s_t = ct.transpose(s_fp16)
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.mma(w, s_t, acc_ws)
        u_f32 = ct.astype(u, ct.float32)
        delta_f32 = u_f32 - w_st
        delta = ct.astype(delta_f32, K.dtype)
        
        # O = Q @ S^T + mask(Q @ K^T) @ delta
        acc_qs = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o1 = ct.mma(q, s_t, acc_qs)
        k_t = ct.transpose(k)
        acc_qk = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, acc_qk)
        qk_masked = qk * mask
        qk_masked_fp16 = ct.astype(qk_masked, K.dtype)
        acc_o2 = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o2 = ct.mma(qk_masked_fp16, delta, acc_o2)
        o_chunk = o1 + o2
        o_out = ct.astype(o_chunk, O.dtype)
        
        # O: strided store (B, T, H, d) -> store (1, chunk_size, 1, d)
        o_out = ct.reshape(o_out, (1, chunk_size, 1, d))
        ct.store(O, index=(b_idx, t_start, h_idx, 0), tile=o_out)
        
        # S = S * gate + delta^T @ K
        # gate is (1, d), broadcasts across rows of S
        delta_t = ct.transpose(delta)
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(delta_t, k, acc_su)
        
        # Element-wise: s * gate, gate (1, d) broadcasts to (d, d)
        gate_f32 = ct.astype(gate, ct.float32)
        s = s * gate_f32 + s_update
    
    s_out = ct.reshape(s, (1, 1, d, d))
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def gated_delta_net_forward(
    S: torch.Tensor,   # (B, H, d, d), float32
    T: torch.Tensor,   # (B, seq_len, H, chunk_size), bfloat16
    G: torch.Tensor,   # (B, num_chunks, H, d), bfloat16 - gate vectors (one per chunk)
    V: torch.Tensor,   # (B, seq_len, H, d), bfloat16
    Q: torch.Tensor,   # (B, seq_len, H, d), bfloat16
    K: torch.Tensor,   # (B, seq_len, H, d), bfloat16
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Gated Delta Net forward pass (batched) with precomputed T and gating.
    
    Input Layout: V, Q, K: (B, T, H, D) - strided load, no reshape
                  G: (B, num_chunks, H, D) - one gate vector per chunk
                  T: (B, seq_len, H, chunk_size) - one vector per timestep
    Output Layout: O: (B, T, H, D) - strided store
    
    Args:
        S: State matrices (B, H, d, d), float32, initialized to zeros
        T: Precomputed T (B, seq_len, H, chunk_size), bfloat16 - one vector per timestep
        G: Gate values (B, num_chunks, H, d), bfloat16 - one gate vector per chunk
        V: Values (B, seq_len, H, d), bfloat16
        Q: Queries (B, seq_len, H, d), bfloat16
        K: Keys (B, seq_len, H, d), bfloat16
        chunk_size: Processing chunk size (64 or 128)
    
    Returns:
        O: Output tensor (B, seq_len, H, d), bfloat16
        (S is updated in-place)
    """
    B_dim, seq_len, H, d = K.shape
    device = K.device
    num_chunks = seq_len // chunk_size
    
    assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    assert S.shape == (B_dim, H, d, d), f"S shape mismatch: expected ({B_dim}, {H}, {d}, {d}), got {S.shape}"
    assert S.dtype == torch.float32, "S must be float32"
    assert T.shape == (B_dim, seq_len, H, chunk_size), f"T shape mismatch: expected ({B_dim}, {seq_len}, {H}, {chunk_size}), got {T.shape}"
    assert G.shape == (B_dim, num_chunks, H, d), f"G shape mismatch: expected ({B_dim}, {num_chunks}, {H}, {d}), got {G.shape}"
    
    # No reshape needed - kernel uses strided loads/stores
    # Output tensor: (B, seq_len, H, d)
    O = torch.empty(B_dim, seq_len, H, d, dtype=K.dtype, device=device)
    
    # Create causal mask (lower triangular) - shared across batches
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device))
    
    # Launch kernel: one block per (B, H) pair
    grid = (B_dim * H,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            gated_delta_net_kernel,
            (S, T, V, Q, K, G, O,
             causal_mask, d, chunk_size, num_chunks, H)
        )
    
    return O


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()
    
    print("--- Fused GDN Benchmark (B, T, H, D) - No Reshape ---")
    
    B_dim = args.batch
    H = args.heads
    seq_len = args.seq_len
    d = HEAD_DIM
    chunk_size = CHUNK_SIZE
    device = 'cuda'
    num_chunks = seq_len // chunk_size
    
    # Initialize inputs with shape (B, T, H, D) - no reshape needed
    S = torch.zeros(B_dim, H, d, d, dtype=torch.float32, device=device)
    V = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    Q = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    K = torch.randn(B_dim, seq_len, H, d, dtype=torch.bfloat16, device=device) * 0.02
    G = torch.rand(B_dim, num_chunks, H, d, dtype=torch.bfloat16, device=device)  # Gate vectors (0-1), one per chunk
    
    print(f"B={B_dim}, H={H}, seq={seq_len}, d={d}, chunks={num_chunks}")
    print(f"Total batch (B*H)={B_dim * H}")
    
    # ============================================================
    # Benchmark: GDN kernel
    # ============================================================
    print("\n[GDN Kernel]")
    
    # Create dummy T tensor (identity-like for testing): (B, seq_len, H, chunk_size)
    # Each chunk forms an identity matrix when loaded
    T = torch.zeros(B_dim, seq_len, H, chunk_size, dtype=torch.bfloat16, device=device)
    for c in range(num_chunks):
        for i in range(chunk_size):
            T[:, c * chunk_size + i, :, i] = 1.0
    
    # Warmup
    for _ in range(3):
        O = gated_delta_net_forward(torch.zeros_like(S), T, G, V, Q, K, chunk_size)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Timing
    start.record()
    for _ in range(args.num_iters):
        O = gated_delta_net_forward(torch.zeros_like(S), T, G, V, Q, K, chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.num_iters
    print(f"  GDN Time: {ms:.3f} ms")

