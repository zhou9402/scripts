# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
cuTile Kernel 4: Fused KDA forward kernel.

This kernel fuses the following operations:
- A_kk_beta = A_kk * beta (column-wise scaling)
- W = A_kk_beta @ K_scaled
- U = A_kk_beta @ V
- v_new = U - W @ S^T
- O = Q_scaled @ S^T + A_qk @ v_new
- S' = S * gk_last_exp + v_new^T @ Kg

Key optimization: A_kk_beta @ V = A_kk @ (V * beta)
This saves one [B, T, H, d] tensor store in the cumsum kernel.
"""

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def fused_kda_kernel_v3(
    S,            # (B, H, d, d) bfloat16 - state (stored as h^T)
    A_qk,         # (B, num_chunks, chunk_size, H, chunk_size) bfloat16 - intra-chunk attention
    A_kk,         # (B, num_chunks, chunk_size, H, chunk_size) bfloat16 - solved A_kk matrix
    K_scaled,     # (B, num_chunks, chunk_size, H, d) bfloat16 - precomputed K * exp2(gk) (no beta!)
    V,            # (B, num_chunks, chunk_size, H, d) bfloat16 - original V (not scaled)
    Kg,           # (B, num_chunks, chunk_size, H, d) bfloat16 - precomputed K * exp2(gk_last - gk)
    Q_scaled,     # (B, num_chunks, chunk_size, H, d) bfloat16 - precomputed Q * exp2(gk) * scale
    Beta,         # (B, num_chunks, chunk_size, H) bfloat16 - beta values
    gk_last_exp,  # (B, num_chunks, H, d) float32 - precomputed exp2(gk_last)
    O,            # (B, num_chunks, chunk_size, H, d) bfloat16 - output
    d: ConstInt, chunk_size: ConstInt, H: ConstInt
):
    """
    Optimized KDA kernel - V3 with beta applied to A_kk instead of precomputing V_scaled.
    
    Key optimization: A_kk_beta @ V = A_kk @ (V * beta)
    This saves one [B, T, H, d] tensor store in the cumsum kernel.
    
    Operations:
    - A_kk_beta = A_kk * beta[None, :] (column-wise scaling)
    - W = A_kk_beta @ K_scaled
    - U = A_kk_beta @ V
    - v_new = U - W @ S^T
    - O = Q_scaled @ S^T + A_qk @ v_new
    - S' = S * gk_last_exp + v_new^T @ Kg
    
    Grid: (B * H,) - one block per (batch, head), sequential across chunks
    """
    num_chunks = A_qk.shape[1]
    
    linear_idx = ct.bid(0)
    b_idx = linear_idx // H
    h_idx = linear_idx % H
    zero_pad = ct.PaddingMode.ZERO
    
    # Load S (stored as h^T), keep as float32 internally
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.astype(ct.reshape(s, (d, d)), ct.float32)
    
    # Causal mask for intra-chunk attention
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)
    
    for c in range(num_chunks):
        # ============================================================
        # Load precomputed values and raw V, Beta
        # ============================================================
        a_qk = ct.reshape(ct.load(A_qk, index=(b_idx, c, 0, h_idx, 0), 
                                  shape=(1, 1, chunk_size, 1, chunk_size), padding_mode=zero_pad), 
                          (chunk_size, chunk_size))
        a_kk = ct.reshape(ct.load(A_kk, index=(b_idx, c, 0, h_idx, 0), 
                                  shape=(1, 1, chunk_size, 1, chunk_size), padding_mode=zero_pad), 
                          (chunk_size, chunk_size))
        
        # Precomputed scaled tensors (K_scaled = K * exp2(gk), no beta)
        k_scaled = ct.reshape(ct.load(K_scaled, index=(b_idx, c, 0, h_idx, 0), 
                                      shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), 
                              (chunk_size, d))
        # Original V (not pre-scaled)
        v_raw = ct.reshape(ct.load(V, index=(b_idx, c, 0, h_idx, 0), 
                                   shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), 
                           (chunk_size, d))
        kg = ct.reshape(ct.load(Kg, index=(b_idx, c, 0, h_idx, 0), 
                                shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), 
                        (chunk_size, d))
        q_scaled = ct.reshape(ct.load(Q_scaled, index=(b_idx, c, 0, h_idx, 0), 
                                      shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), 
                              (chunk_size, d))
        
        # Beta: [B, num_chunks, chunk_size, H] -> [1, chunk_size]
        beta = ct.reshape(ct.load(Beta, index=(b_idx, c, 0, h_idx), 
                                  shape=(1, 1, chunk_size, 1), padding_mode=zero_pad), 
                          (1, chunk_size))
        
        # gk_last_exp: [B, num_chunks, H, d] -> [1, d]
        gk_last_e = ct.reshape(ct.load(gk_last_exp, index=(b_idx, c, h_idx, 0), 
                                       shape=(1, 1, 1, d), padding_mode=zero_pad), 
                               (1, d))
        
        # ============================================================
        # Compute A_kk_beta = A_kk * beta (column-wise scaling)
        # A_kk_beta @ X = A_kk @ (X * beta[:, None])
        # ============================================================
        a_kk_beta = ct.astype(a_kk * beta, K_scaled.dtype)  # [chunk_size, chunk_size]
        
        # Clean A_qk (mask NaN in upper triangular)
        a_qk_clean = ct.astype(ct.where(mask > 0.0, a_qk, 0.0), K_scaled.dtype)
        
        # State transpose
        s_bf16 = ct.astype(s, K_scaled.dtype)
        s_t = ct.transpose(s_bf16)
        
        # W = A_kk_beta @ K_scaled (K_scaled = K * exp2(gk), so this gives K * beta * exp2(gk))
        w = ct.mma(a_kk_beta, k_scaled, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        
        # U = A_kk_beta @ V (this gives V * beta)
        u = ct.mma(a_kk_beta, v_raw, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        
        # W @ S^T
        w_st = ct.mma(ct.astype(w, K_scaled.dtype), s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        
        # v_new = U - W @ S^T
        new_v = u - w_st
        new_v_bf16 = ct.astype(new_v, K_scaled.dtype)
        
        # O_inter = Q_scaled @ S^T (Q_scaled already has scale applied)
        o_inter = ct.mma(q_scaled, s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        
        # O_intra = A_qk @ new_v
        o_intra = ct.mma(a_qk_clean, new_v_bf16, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        
        # Output
        o_out = ct.astype(o_inter + o_intra, O.dtype)
        ct.store(O, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(o_out, (1, 1, chunk_size, 1, d)))
        
        # ============================================================
        # State update: S' = S * gk_last_exp + new_v^T @ Kg
        # ============================================================
        s_update = ct.mma(ct.transpose(new_v_bf16), kg, ct.full((d, d), 0.0, dtype=ct.float32))
        s_decay = s * gk_last_e  # [d, d] * [1, d] -> broadcast
        s = s_decay + s_update
    
    # Store final state
    s_out = ct.astype(ct.reshape(s, (1, 1, d, d)), S.dtype)
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def launch_kda_kernel(S, A_qk, A_kk, K_scaled, V, Kg, Q_scaled, Beta, gk_last_exp, chunk_size):
    """Launch the fused KDA kernel."""
    B, num_chunks, cs, H, d = K_scaled.shape
    
    O_r = torch.empty(B, num_chunks, cs, H, d, dtype=K_scaled.dtype, device=K_scaled.device)
    
    with torch.cuda.device(K_scaled.device):
        ct.launch(torch.cuda.current_stream(), (B * H,), fused_kda_kernel_v3,
                  (S, A_qk, A_kk, K_scaled, V, Kg, Q_scaled, Beta, gk_last_exp, O_r, 
                   d, chunk_size, H))
    
    return O_r
