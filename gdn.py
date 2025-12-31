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


@ct.kernel
def gated_delta_net_kernel(
    S,      # (batch, d, d) float32 - state
    W,      # (batch, num_chunks, chunk_size, d) float16
    U,      # (batch, num_chunks, chunk_size, d) float16
    Q,      # (batch, num_chunks, chunk_size, d) float16
    K,      # (batch, num_chunks, chunk_size, d) float16
    O,      # (batch, num_chunks, chunk_size, d) float16
    causal_mask,  # (chunk_size, chunk_size) float32 - precomputed lower triangular
    d: ConstInt,           # head_dim = 128
    chunk_size: ConstInt,  # 64 or 128
    num_chunks: ConstInt   # number of chunks to process
):
    """
    Fused Gated Delta Rule forward pass.
    
    Each block processes one batch, iterating through all chunks sequentially.
    Grid: (batch,)
    
    For each chunk c:
      1. Load W[batch, c], U[batch, c], Q[batch, c], K[batch, c]
      2. Compute delta = U - W @ S^T
      3. Compute O[batch, c] = Q @ S^T + mask(Q @ K^T) @ delta
      4. Update S = S + delta^T @ K
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
        # Step 1: Load current chunk data with TMA and latency hints
        # ============================================================
        w = ct.load(W, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        w = ct.reshape(w, (chunk_size, d))
        
        u = ct.load(U, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        u = ct.reshape(u, (chunk_size, d))
        
        q = ct.load(Q, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        q = ct.reshape(q, (chunk_size, d))
        
        k = ct.load(K, index=(batch_idx, c, 0, 0), shape=(1, 1, chunk_size, d), 
                    padding_mode=zero_pad, allow_tma=True, latency=3)
        k = ct.reshape(k, (chunk_size, d))
        
        # ============================================================
        # Step 2: Convert S to input dtype for MMA (inputs fp16, accumulate fp32)
        # ============================================================
        s_fp16 = ct.astype(s, W.dtype)      # [d, d] -> fp16
        s_t = ct.transpose(s_fp16)           # [d, d] transposed
        
        # ============================================================
        # Step 3: Compute delta = U - W @ S^T  [chunk, d]
        #         This is reused for both O and S update
        # ============================================================
        # W @ S^T: [chunk, d] @ [d, d] = [chunk, d], fp16 inputs, fp32 accum
        acc_ws = ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        w_st = ct.mma(w, s_t, acc_ws)
        
        # delta = U - W @ S^T (in fp32, then convert to fp16 for next MMA)
        u_f32 = ct.astype(u, ct.float32)
        delta_f32 = u_f32 - w_st  # [chunk, d] in fp32
        delta = ct.astype(delta_f32, W.dtype)  # [chunk, d] in fp16
        
        # ============================================================
        # Step 4: Compute O = Q @ S^T + mask(Q @ K^T) @ delta
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
        qk_masked_fp16 = ct.astype(qk_masked, W.dtype)
        
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
        # Step 5: Update state S = S + delta^T @ K
        #         This can overlap with next iteration's loads
        # ============================================================
        # delta^T @ K: [d, chunk] @ [chunk, d] = [d, d], fp16 inputs, fp32 accum
        delta_t = ct.transpose(delta)  # [d, chunk] in fp16
        acc_su = ct.full((d, d), 0.0, dtype=ct.float32)
        s_update = ct.mma(delta_t, k, acc_su)  # fp16 inputs, fp32 accum
        
        # S = S + delta^T @ K
        s = s + s_update
    
    # Store final state S for this batch
    s_out = ct.reshape(s, (1, d, d))
    ct.store(S, index=(batch_idx, 0, 0), tile=s_out)


def gated_delta_net_forward(
    S: torch.Tensor,   # (batch, d, d), float32
    W: torch.Tensor,   # (batch, seq_len, d), float16
    U: torch.Tensor,   # (batch, seq_len, d), float16
    Q: torch.Tensor,   # (batch, seq_len, d), float16
    K: torch.Tensor,   # (batch, seq_len, d), float16
    chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """
    Gated Delta Net forward pass (batched).
    
    Args:
        S: State matrices (batch, d, d), float32, initialized to zeros
        W: Gate weights (batch, seq_len, d), float16
        U: Update values (batch, seq_len, d), float16
        Q: Queries (batch, seq_len, d), float16
        K: Keys (batch, seq_len, d), float16
        chunk_size: Processing chunk size (64 or 128)
    
    Returns:
        O: Output tensor (batch, seq_len, d), float16
        (S is updated in-place)
    """
    batch, seq_len, d = W.shape
    device = W.device
    
    assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    assert S.shape == (batch, d, d), f"S shape mismatch: expected ({batch}, {d}, {d}), got {S.shape}"
    assert S.dtype == torch.float32, "S must be float32"
    
    num_chunks = seq_len // chunk_size
    
    # Reshape inputs for tile indexing: (batch, num_chunks, chunk_size, d)
    W_reshaped = W.reshape(batch, num_chunks, chunk_size, d).contiguous()
    U_reshaped = U.reshape(batch, num_chunks, chunk_size, d).contiguous()
    Q_reshaped = Q.reshape(batch, num_chunks, chunk_size, d).contiguous()
    K_reshaped = K.reshape(batch, num_chunks, chunk_size, d).contiguous()
    
    # Output tensor
    O = torch.empty(batch, num_chunks, chunk_size, d, dtype=W.dtype, device=device)
    
    # Create causal mask (lower triangular) - shared across batches
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device))
    
    # Launch kernel: one block per batch
    grid = (batch,)
    
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            gated_delta_net_kernel,
            (S, W_reshaped, U_reshaped, Q_reshaped, K_reshaped, O, 
             causal_mask, d, chunk_size, num_chunks)
        )
    
    # Reshape output back to (batch, seq_len, d)
    return O.reshape(batch, seq_len, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    
    print("--- Gated Delta Net: Fused chunk_gated_delta_rule_fwd_h + chunk_fwd_o ---")
    
    batch = args.batch
    seq_len = args.seq_len
    d = args.head_dim
    chunk_size = args.chunk_size
    device = 'cuda'
    
    # Initialize inputs
    S = torch.zeros(batch, d, d, dtype=torch.float32, device=device)
    W = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    U = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    Q = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    K = torch.randn(batch, seq_len, d, dtype=torch.float16, device=device) * 0.02
    
    print(f"Configuration:")
    print(f"  batch: {batch}")
    print(f"  seq_len: {seq_len}")
    print(f"  head_dim: {d}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  num_chunks: {seq_len // chunk_size}")
    print(f"  grid: ({batch},)")
    print(f"  S: {S.shape}, dtype: {S.dtype}")
    print(f"  W/U/Q/K: {W.shape}, dtype: {W.dtype}")
    
    # Warmup
    S_warmup = S.clone()
    O = gated_delta_net_forward(S_warmup, W, U, Q, K, chunk_size)
    torch.cuda.synchronize()
    
    print(f"\nOutput O: {O.shape}, dtype: {O.dtype}")
    
    # Timing with CUDA events
    print("\n--- Performance ---")
    num_iters = 100
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        S_bench = torch.zeros(batch, d, d, dtype=torch.float32, device=device)
        O = gated_delta_net_forward(S_bench, W, U, Q, K, chunk_size)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iters
    
    # ============================================================
    # Calculate TFLOPS
    # ============================================================
    # Per chunk per batch:
    #   W @ S^T: 2 * chunk * d * d
    #   Q @ S^T: 2 * chunk * d * d
    #   Q @ K^T: 2 * chunk * d * chunk
    #   mask @ delta: 2 * chunk * chunk * d
    #   delta^T @ K: 2 * d * chunk * d
    num_chunks = seq_len // chunk_size
    flops_per_chunk = (
        2 * chunk_size * d * d +         # W @ S^T
        2 * chunk_size * d * d +         # Q @ S^T
        2 * chunk_size * d * chunk_size + # Q @ K^T
        2 * chunk_size * chunk_size * d + # qk_masked @ delta
        2 * d * chunk_size * d           # delta^T @ K
    )
    total_flops = batch * num_chunks * flops_per_chunk
    tflops = total_flops / (avg_ms * 1e-3) / 1e12
    
    # ============================================================
    # Calculate Bandwidth
    # ============================================================
    # Per chunk per batch read:
    #   W: chunk * d * 2 bytes (fp16)
    #   U: chunk * d * 2 bytes (fp16)
    #   Q: chunk * d * 2 bytes (fp16)
    #   K: chunk * d * 2 bytes (fp16)
    # Per chunk per batch write:
    #   O: chunk * d * 2 bytes (fp16)
    # Per batch one-time:
    #   S read: d * d * 4 bytes (fp32)
    #   S write: d * d * 4 bytes (fp32)
    # Shared:
    #   causal_mask: chunk * chunk * 4 bytes (fp32) - read once, cached
    
    bytes_per_chunk = 4 * chunk_size * d * 2 + chunk_size * d * 2  # read W,U,Q,K + write O
    bytes_per_batch = (
        num_chunks * bytes_per_chunk +   # per-chunk IO
        2 * d * d * 4                    # S read + write (fp32)
    )
    bytes_total = batch * bytes_per_batch + chunk_size * chunk_size * 4  # + causal_mask
    bandwidth_gb_s = bytes_total / (avg_ms * 1e-3) / 1e9
    
    print(f"Average time: {avg_ms:.3f} ms ({num_iters} iterations)")
    print(f"TFLOPS: {tflops:.2f}")
    print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"Data transferred: {bytes_total / 1e6:.2f} MB")
    
    print("\n--- Done ---")

