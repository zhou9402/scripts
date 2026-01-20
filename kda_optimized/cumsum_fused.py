# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Triton Kernel 1: Fused chunk_local_cumsum with scaling for KDA.

This kernel computes cumsum and all scaling operations in a single pass:
- G_cumsum = cumsum(G) * scale
- K_scaled = K * exp2(G_cumsum)  (beta is applied in cuTile kernel to A_kk)
- Kg = K * exp2(gk_last - G_cumsum)
- Q_scaled = Q * exp2(G_cumsum) * attn_scale
- gk_last_exp = exp2(gk_last)

Note: V_scaled is NOT computed here. Instead, beta is applied to A_kk in the
cuTile kernel: A_kk_beta @ V = A_kk @ (V * beta). This saves one tensor store.
"""

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'HAS_SCALE': lambda args: args['cumsum_scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_with_scaling_kernel(
    # Inputs
    g,              # [B, T, H, S] - gate (log space)
    k,              # [B, T, H, S] - keys
    q,              # [B, T, H, S] - queries
    cumsum_scale,   # float - scale for cumsum (RCP_LN2)
    attn_scale,     # float - attention scale
    # Outputs
    g_cumsum,       # [B, T, H, S] - cumsum result
    k_scaled,       # [B, T, H, S] - K * exp2(gk) (beta moved to A_kk in cuTile)
    kg,             # [B, T, H, S] - K * exp2(gk_last - gk)
    q_scaled,       # [B, T, H, S] - Q * exp2(gk) * attn_scale
    gk_last_exp,    # [B, num_chunks, H, S] - exp2(gk_last)
    # Index info
    cu_seqlens,
    chunk_indices,
    T,
    # Constants
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: cumsum + all scaling operations for KDA.
    
    Grid: (cdiv(S, BS), NT, B * H)
    """
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    # Compute chunk boundaries
    chunk_start = i_t * BT
    chunk_end = min(chunk_start + BT, T)
    last_idx = chunk_end - 1 - chunk_start  # Relative index within chunk
    
    # Pointers for g, k, q (all [B, T, H, S])
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    
    # Output pointers
    p_g_out = tl.make_block_ptr(g_cumsum + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_ks = tl.make_block_ptr(k_scaled + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_qs = tl.make_block_ptr(q_scaled + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    
    # Load inputs [BT, BS]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    
    # Step 1: Compute cumsum
    b_gk = tl.cumsum(b_g, axis=0)
    if HAS_SCALE:
        b_gk = b_gk * cumsum_scale
    
    # Store cumsum result
    tl.store(p_g_out, b_gk.to(p_g_out.dtype.element_ty), boundary_check=(0, 1))
    
    # Step 2: Get gk_last [1, BS] - last row of cumsum within this chunk
    o_t = tl.arange(0, BT)
    m_last = o_t == last_idx
    b_gk_last = tl.sum(tl.where(m_last[:, None], b_gk, 0.0), axis=0, keep_dims=True)  # [1, BS]
    
    # Step 3: Compute exp2 values
    b_exp2_gk = exp2(b_gk)                    # [BT, BS]
    b_exp2_kg = exp2(b_gk_last - b_gk)        # [BT, BS] - broadcast
    b_exp2_last = exp2(b_gk_last)             # [1, BS]
    
    # Step 4: Compute all scaled outputs
    # K_scaled = K * exp2(gk) (beta is applied to A_kk in cuTile kernel)
    b_ks = b_k * b_exp2_gk
    tl.store(p_ks, b_ks.to(p_ks.dtype.element_ty), boundary_check=(0, 1))
    
    # Note: V_scaled is NOT computed here. Beta is applied to A_kk in cuTile kernel.
    
    # Kg = K * exp2(gk_last - gk)
    b_kg = b_k * b_exp2_kg
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
    
    # Q_scaled = Q * exp2(gk) * attn_scale
    b_qs = b_q * b_exp2_gk * attn_scale
    tl.store(p_qs, b_qs.to(p_qs.dtype.element_ty), boundary_check=(0, 1))
    
    # Step 5: Store gk_last_exp [B, num_chunks, H, S]
    NT = tl.cdiv(T, BT)
    p_gk_last_exp = tl.make_block_ptr(
        gk_last_exp + (i_b * NT * H + i_t * H + i_h) * S,
        (S,), (1,), (i_s * BS,), (BS,), (0,)
    )
    tl.store(p_gk_last_exp, b_exp2_last.reshape((BS,)).to(p_gk_last_exp.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_with_scaling(
    g: torch.Tensor,           # [B, T, H, S] - gate
    k: torch.Tensor,           # [B, T, H, S] - keys
    q: torch.Tensor,           # [B, T, H, S] - queries
    chunk_size: int,
    cumsum_scale: float = None,    # RCP_LN2
    attn_scale: float = 1.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused cumsum with KDA scaling operations.
    
    Note: V_scaled is NOT computed here. Beta is applied to A_kk in the cuTile kernel.
    This saves one tensor store: A_kk_beta @ V = A_kk @ (V * beta).
    
    Args:
        g: Gate tensor [B, T, H, S] (log space)
        k: Key tensor [B, T, H, S]
        q: Query tensor [B, T, H, S]
        chunk_size: Chunk size (BT)
        cumsum_scale: Scale for cumsum (typically RCP_LN2)
        attn_scale: Attention scale (typically 1/sqrt(d))
        cu_seqlens: Cumulative sequence lengths for variable length
        chunk_indices: Chunk indices for variable length
    
    Returns:
        g_cumsum: [B, T, H, S] - cumsum result
        k_scaled: [B, T, H, S] - K * exp2(gk) (beta applied in cuTile)
        kg: [B, T, H, S] - K * exp2(gk_last - gk)
        q_scaled: [B, T, H, S] - Q * exp2(gk) * attn_scale
        gk_last_exp: [B, num_chunks, H, S] - exp2(gk_last)
    """
    B, T, H, S = g.shape
    BT = chunk_size
    
    assert T % BT == 0, f"T ({T}) must be divisible by chunk_size ({BT})"
    
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    # Allocate outputs (no v_scaled - saves memory!)
    g_cumsum = torch.empty_like(g, dtype=torch.float32)
    k_scaled = torch.empty_like(k)
    kg = torch.empty_like(k)
    q_scaled = torch.empty_like(q)
    gk_last_exp = torch.empty(B, NT, H, S, dtype=torch.float32, device=g.device)
    
    def grid(meta):
        return (triton.cdiv(S, meta['BS']), NT, B * H)
    
    chunk_local_cumsum_with_scaling_kernel[grid](
        g=g,
        k=k,
        q=q,
        cumsum_scale=cumsum_scale,
        attn_scale=attn_scale,
        g_cumsum=g_cumsum,
        k_scaled=k_scaled,
        kg=kg,
        q_scaled=q_scaled,
        gk_last_exp=gk_last_exp,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
    )
    
    return g_cumsum, k_scaled, kg, q_scaled, gk_last_exp
