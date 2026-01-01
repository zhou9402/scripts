
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated Delta Net: Fused chunk_gated_delta_rule_fwd_h and chunk_fwd_o

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

Inputs:
  S: (batch, d, d) float32 - state matrices (initialized to 0)
  W: (batch, seq_len, d) float16 - gate weights
  U: (batch, seq_len, d) float16 - update values
  Q: (batch, seq_len, d) float16 - queries
  K: (batch, seq_len, d) float16 - keys
  
Outputs:
  O: (batch, seq_len, d) float16 - output
  S: (batch, d, d) float32 - updated states (in-place)
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


@ct.kernel
def compute_T_kernel(
    K,      # (batch, num_chunks, chunk_size, d) float16
    B,      # (batch, num_chunks, chunk_size) float16 - beta values
    T_out,  # (batch, num_chunks, chunk_size, chunk_size) float16 - output
    d: ConstInt,
    chunk_size: ConstInt,
    num_chunks: ConstInt
):
    """
    Compute T = (I + A)^{-1} for all chunks in parallel.
    
    Grid: (batch * num_chunks,)
    Each block computes one (batch, chunk) pair.
    
    Steps:
      1. Load K chunk, load beta chunk
      2. A = (K * diag(beta)) @ K^T, take strictly lower triangular
      3. T = (I + A)^{-1}
      4. Store T
    """
    linear_idx = ct.bid(0)
    batch_idx = linear_idx // num_chunks
    chunk_idx = linear_idx % num_chunks
    zero_pad = ct.PaddingMode.ZERO
    
    # 16x16 identity for solve (fp16)
    idx16 = ct.arange(16, dtype=ct.int32)
    r16 = ct.reshape(idx16, (16, 1))
    c16 = ct.reshape(idx16, (1, 16))
    identity_16 = ct.where(r16 == c16, ct.full((16, 16), 1.0, dtype=K.dtype),
                          ct.full((16, 16), 0.0, dtype=K.dtype))
    
    # 64x64 strictly lower triangular mask
    idx64 = ct.arange(chunk_size, dtype=ct.int32)
    r64 = ct.reshape(idx64, (chunk_size, 1))
    c64 = ct.reshape(idx64, (1, chunk_size))
    
    # ============================================================
    # Step 1: Load K and Beta for this chunk
    # ============================================================
    k = ct.load(K, index=(batch_idx, chunk_idx, 0, 0), shape=(1, 1, chunk_size, d), 
                padding_mode=zero_pad, allow_tma=True)
    k = ct.reshape(k, (chunk_size, d))  # [chunk_size, d]
    
    beta = ct.load(B, index=(batch_idx, chunk_idx, 0), shape=(1, 1, chunk_size), 
                   padding_mode=zero_pad, allow_tma=True)
    beta = ct.reshape(beta, (chunk_size, 1))  # [chunk_size, 1] for broadcasting
    
    # ============================================================
    # Step 2: A = (K * diag(beta)) @ K^T, strictly lower triangular
    # ============================================================
    # K * diag(beta) = K * beta (broadcast along d dimension)
    k_beta = k * beta  # [chunk_size, d], element-wise: each row scaled by its beta
    
    # A = k_beta @ K^T: [chunk, d] @ [d, chunk] = [chunk, chunk]
    k_t = ct.transpose(k)
    acc_kkt = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
    A_full = ct.mma(k_beta, k_t, acc_kkt)
    
    # Convert to fp16 first, then take strictly lower triangular
    A_fp16 = ct.astype(A_full, K.dtype)
    A_tile = ct.where(r64 > c64, A_fp16, ct.full((chunk_size, chunk_size), 0.0, dtype=K.dtype))
    
    # ============================================================
    # Step 3: Solve T = (I + A)^{-1}
    # ============================================================
    # Extract 16x16 blocks from A
    a11 = ct.extract(A_tile, index=(0, 0), shape=(16, 16))
    a22 = ct.extract(A_tile, index=(16, 16), shape=(16, 16))
    a33 = ct.extract(A_tile, index=(32, 32), shape=(16, 16))
    a44 = ct.extract(A_tile, index=(48, 48), shape=(16, 16))
    a21 = ct.extract(A_tile, index=(16, 0), shape=(16, 16))
    a31 = ct.extract(A_tile, index=(32, 0), shape=(16, 16))
    a32 = ct.extract(A_tile, index=(32, 16), shape=(16, 16))
    a41 = ct.extract(A_tile, index=(48, 0), shape=(16, 16))
    a42 = ct.extract(A_tile, index=(48, 16), shape=(16, 16))
    a43 = ct.extract(A_tile, index=(48, 32), shape=(16, 16))
    
    # Solve diagonal blocks
    t11 = solve_16x16_block(a11, identity_16, K.dtype)
    t22 = solve_16x16_block(a22, identity_16, K.dtype)
    t33 = solve_16x16_block(a33, identity_16, K.dtype)
    t44 = solve_16x16_block(a44, identity_16, K.dtype)
    
    t11_fp16 = ct.astype(t11, K.dtype)
    t22_fp16 = ct.astype(t22, K.dtype)
    t33_fp16 = ct.astype(t33, K.dtype)
    t44_fp16 = ct.astype(t44, K.dtype)
    
    # Off-diagonal blocks
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a21, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t21_fp16 = ct.astype(-ct.mma(t22_fp16, ct.astype(tmp, K.dtype), acc), K.dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a32, t22_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t32_fp16 = ct.astype(-ct.mma(t33_fp16, ct.astype(tmp, K.dtype), acc), K.dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp = ct.mma(a43, t33_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t43_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp, K.dtype), acc), K.dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a31, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a32, t21_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t31_fp16 = ct.astype(-ct.mma(t33_fp16, ct.astype(tmp1 + tmp2, K.dtype), acc), K.dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a42, t22_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a43, t32_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t42_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp1 + tmp2, K.dtype), acc), K.dtype)
    
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp1 = ct.mma(a41, t11_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp2 = ct.mma(a42, t21_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    tmp3 = ct.mma(a43, t31_fp16, acc)
    acc = ct.full((16, 16), 0.0, dtype=ct.float32)
    t41_fp16 = ct.astype(-ct.mma(t44_fp16, ct.astype(tmp1 + tmp2 + tmp3, K.dtype), acc), K.dtype)
    
    # Assemble T using cat
    zero16 = ct.full((16, 16), 0.0, dtype=K.dtype)
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
    
    # ============================================================
    # Step 4: Store T
    # ============================================================
    T_reshaped = ct.reshape(T, (1, 1, chunk_size, chunk_size))
    ct.store(T_out, index=(batch_idx, chunk_idx, 0, 0), tile=T_reshaped, allow_tma=True)


@ct.kernel
def gated_delta_net_kernel(
    S,      # (batch, d, d) float32 - state
    T_in,   # (batch, num_chunks, chunk_size, chunk_size) float16 - precomputed T
    V,      # (batch, num_chunks, chunk_size, d) float16
    Q,      # (batch, num_chunks, chunk_size, d) float16
    K,      # (batch, num_chunks, chunk_size, d) float16
    G,      # (batch, num_chunks) float16 - gate values (one scalar per chunk)
    O,      # (batch, num_chunks, chunk_size, d) float16
    causal_mask,  # (chunk_size, chunk_size) float32 - precomputed lower triangular
    d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    num_chunks: ConstInt   # number of chunks to process
):
    """
    Fused Gated Delta Rule forward pass (T precomputed) with gating.
    
    Each block processes one batch, iterating through all chunks sequentially.
    Grid: (batch,)
    
    For each chunk c:
      1. Load T, K, V, Q, gate
      2. Compute W = T @ K, U = T @ V
      3. Compute delta = U - W @ S^T
      4. Compute O[batch, c] = Q @ S^T + mask(Q @ K^T) @ delta
      5. Update S = S * gate + delta^T @ K
    """
    batch_idx = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO
    
    # Load initial state S for this batch: (d, d) in float32
    s = ct.load(S, index=(batch_idx, 0, 0), shape=(1, d, d), padding_mode=zero_pad)
    s = ct.reshape(s, (d, d))
    
    # Load causal mask: (chunk_size, chunk_size) - shared across batches
    mask = ct.load(causal_mask, index=(0, 0), shape=(chunk_size, chunk_size), padding_mode=zero_pad)
    
    # Process each chunk
    for c in range(num_chunks):
        # ============================================================
        # Step 1: Load T, K, V, Q, gate
        # ============================================================
        T = ct.load(T_in, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, chunk_size), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        T = ct.reshape(T, (chunk_size, chunk_size))
        
        k = ct.load(K, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        k = ct.reshape(k, (chunk_size, d))
        
        # Load gate scalar for this chunk
        gate = ct.load(G, index=(batch_idx, c), shape=(1, 1), padding_mode=zero_pad)
        gate = ct.reshape(gate, (1, 1))  # scalar for broadcasting
        
        # ============================================================
        # Step 2: Load V, Q
        # ============================================================
        v = ct.load(V, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        v = ct.reshape(v, (chunk_size, d))
        
        q = ct.load(Q, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        q = ct.reshape(q, (chunk_size, d))
        
        # ============================================================
        # Step 4: Compute W = T @ K, U = T @ V
        # ============================================================
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w = ct.astype(ct.mma(T, k, acc), K.dtype)
        
        acc = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        u = ct.astype(ct.mma(T, v, acc), K.dtype)
        
        # ============================================================
        # Step 5: Convert S to input dtype for MMA (inputs fp16, accumulate fp32)
        # ============================================================
        s_fp16 = ct.astype(s, K.dtype)      # [d, d] -> fp16
        s_t = ct.transpose(s_fp16)           # [d, d] transposed
        
        # ============================================================
        # Step 6: Compute delta = U - W @ S^T  [chunk, d]
        #         This is reused for both O and S update
        # ============================================================
        # W @ S^T: [chunk, d] @ [d, d] = [chunk, d], fp16 inputs, fp32 accum
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.mma(w, s_t, acc_ws)
        
        # delta = U - W @ S^T (in fp32, then convert to fp16 for next MMA)
        u_f32 = ct.astype(u, ct.float32)
        delta_f32 = u_f32 - w_st  # [chunk, d] in fp32
        delta = ct.astype(delta_f32, K.dtype)  # [chunk, d] in fp16
        
        # ============================================================
        # Step 5: Compute O = Q @ S^T + mask(Q @ K^T) @ delta
        #         Uses current S (before update)
        # ============================================================
        
        # Part 1: Q @ S^T  [chunk, d] @ [d, d] = [chunk, d]
        acc_qs = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o1 = ct.mma(q, s_t, acc_qs)  # fp16 inputs, fp32 accum
        
        # Part 2: mask(Q @ K^T) @ delta
        # Q @ K^T: [chunk, d] @ [d, chunk] = [chunk, chunk]
        k_t = ct.transpose(k)  # [d, chunk]
        acc_qk = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, acc_qk)  # [chunk, chunk], fp32
        
        # Apply causal mask (element-wise multiply with lower triangular mask)
        qk_masked = qk * mask  # [chunk, chunk], fp32
        
        # Convert to fp16 for next MMA
        qk_masked_fp16 = ct.astype(qk_masked, K.dtype)
        
        # qk_masked @ delta: [chunk, chunk] @ [chunk, d] = [chunk, d]
        acc_o2 = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        o2 = ct.mma(qk_masked_fp16, delta, acc_o2)  # fp16 inputs, fp32 accum
        
        # O = o1 + o2 (in fp32)
        o_chunk = o1 + o2  # [chunk, d] in fp32
        
        # Store O chunk with TMA
        o_out = ct.astype(o_chunk, O.dtype)
        o_out = ct.reshape(o_out, (1, 1, chunk_size, d))
        ct.store(O, index=(batch_idx, c, 0, 0), tile=o_out, allow_tma=True)
        
        # ============================================================
        # Step 6: Update state S = S * gate + delta^T @ K
        #         This can overlap with next iteration's loads
        # ============================================================
        # delta^T @ K: [d, chunk] @ [chunk, d] = [d, d], fp16 inputs, fp32 accum
        delta_t = ct.transpose(delta)  # [d, chunk] in fp16
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(delta_t, k, acc_su)  # fp16 inputs, fp32 accum
        
        # S = S * gate + delta^T @ K (gate is fp16 scalar, broadcast to all elements)
        gate_f32 = ct.astype(gate, ct.float32)
        s = s * gate_f32 + s_update
    
    # Store final state S for this batch
    s_out = ct.reshape(s, (1, d, d))
    ct.store(S, index=(batch_idx, 0, 0), tile=s_out)


def compute_T_forward(
    K: torch.Tensor,   # (batch, seq_len, d), float16
    B: torch.Tensor,   # (batch, seq_len), float16 - beta values
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Compute T = (I + A)^{-1} for all chunks in parallel.
    
    A = (K * diag(beta)) @ K^T, strictly lower triangular.
    
    Args:
        K: Keys (batch, seq_len, d), float16
        B: Beta values (batch, seq_len), float16
        chunk_size: Processing chunk size (64)
    
    Returns:
        T: Precomputed inverse matrices (batch, num_chunks, chunk_size, chunk_size), float16
    """
    batch, seq_len, d = K.shape
    device = K.device
    num_chunks = seq_len // chunk_size
    
    assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    assert B.shape == (batch, seq_len), f"B shape mismatch: expected ({batch}, {seq_len}), got {B.shape}"
    
    # Reshape inputs for tile indexing
    K_reshaped = K.reshape(batch, num_chunks, chunk_size, d).contiguous()
    B_reshaped = B.reshape(batch, num_chunks, chunk_size).contiguous()
    
    # Output tensor
    T = torch.empty(batch, num_chunks, chunk_size, chunk_size, dtype=K.dtype, device=device)
    
    # Launch kernel: one block per (batch, chunk) pair
    grid = (batch * num_chunks,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            compute_T_kernel,
            (K_reshaped, B_reshaped, T, d, chunk_size, num_chunks)
        )
    
    return T


def gated_delta_net_forward(
    S: torch.Tensor,   # (batch, d, d), float32
    T: torch.Tensor,   # (batch, num_chunks, chunk_size, chunk_size), float16
    G: torch.Tensor,   # (batch, num_chunks), float16 - gate values
    V: torch.Tensor,   # (batch, seq_len, d), float16
    Q: torch.Tensor,   # (batch, seq_len, d), float16
    K: torch.Tensor,   # (batch, seq_len, d), float16
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Gated Delta Net forward pass (batched) with precomputed T and gating.
    
    Args:
        S: State matrices (batch, d, d), float32, initialized to zeros
        T: Precomputed T = (I + A)^{-1} (batch, num_chunks, chunk_size, chunk_size), float16
        G: Gate values (batch, num_chunks), float16 - one scalar per chunk
        V: Values (batch, seq_len, d), float16
        Q: Queries (batch, seq_len, d), float16
        K: Keys (batch, seq_len, d), float16
        chunk_size: Processing chunk size (64 or 128)
    
    Returns:
        O: Output tensor (batch, seq_len, d), float16
        (S is updated in-place)
    """
    batch, seq_len, d = K.shape
    device = K.device
    num_chunks = seq_len // chunk_size
    
    assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    assert S.shape == (batch, d, d), f"S shape mismatch: expected ({batch}, {d}, {d}), got {S.shape}"
    assert S.dtype == torch.float32, "S must be float32"
    assert T.shape == (batch, num_chunks, chunk_size, chunk_size), f"T shape mismatch"
    assert G.shape == (batch, num_chunks), f"G shape mismatch: expected ({batch}, {num_chunks}), got {G.shape}"
    
    # Reshape inputs for tile indexing: (batch, num_chunks, chunk_size, d)
    V_reshaped = V.reshape(batch, num_chunks, chunk_size, d).contiguous()
    Q_reshaped = Q.reshape(batch, num_chunks, chunk_size, d).contiguous()
    K_reshaped = K.reshape(batch, num_chunks, chunk_size, d).contiguous()
    
    # Output tensor
    O = torch.empty(batch, num_chunks, chunk_size, d, dtype=K.dtype, device=device)
    
    # Create causal mask (lower triangular) - shared across batches
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device))
    
    # Launch kernel: one block per batch
    grid = (batch,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            gated_delta_net_kernel,
            (S, T, V_reshaped, Q_reshaped, K_reshaped, G, O,
             causal_mask, d, chunk_size, num_chunks)
        )
    
    # Reshape output back to (batch, seq_len, d)
    return O.reshape(batch, seq_len, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()
    
    print("--- Fused GDN (Two-Stage: compute_T + GDN) ---")
    
    batch = args.batch
    seq_len = args.seq_len
    d = HEAD_DIM
    chunk_size = CHUNK_SIZE
    device = 'cuda'
    num_chunks = seq_len // chunk_size
    
    # Initialize inputs
    S = torch.zeros(batch, d, d, dtype=torch.float32, device=device)
    V = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    Q = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    K = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    B = torch.randn(batch, seq_len, dtype=torch.float16, device=device) * 0.1  # Beta values
    G = torch.rand(batch, num_chunks, dtype=torch.float16, device=device)     # Gate values (0-1)
    
    print(f"batch={batch}, seq={seq_len}, d={d}, chunks={num_chunks}")
    
    # Warmup
    for _ in range(3):
        T = compute_T_forward(K, B, chunk_size)
        O = gated_delta_net_forward(torch.zeros_like(S), T, G, V, Q, K, chunk_size)
    torch.cuda.synchronize()
    
    print(f"T shape: {T.shape}, dtype: {T.dtype}")
    print(f"Output O: {O.shape}, dtype: {O.dtype}")
    
    # Timing - Stage 1: compute_T
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.num_iters):
        T = compute_T_forward(K, B, chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_T = start.elapsed_time(end) / args.num_iters
    print(f"compute_T Time: {ms_T:.3f} ms")
    
    # Timing - Stage 2: GDN
    start.record()
    for _ in range(args.num_iters):
        O = gated_delta_net_forward(torch.zeros_like(S), T, G, V, Q, K, chunk_size)
    end.record()
    torch.cuda.synchronize()
    ms_GDN = start.elapsed_time(end) / args.num_iters
    print(f"GDN Time: {ms_GDN:.3f} ms")
    
    # Total
    print(f"Total Time: {ms_T + ms_GDN:.3f} ms")

