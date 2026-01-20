# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
KDA Forward Pass - Optimized Fused Implementation

This module combines all 4 kernels into a single forward pass:
1. cumsum_fused.py     - Triton: cumsum + all scaling operations
2. K2: chunk_kda_fwd_kernel_intra_sub_chunk - Triton: A_qk, A_kkd computation (NEW optimized, safe_gate=True)
3. K3: chunk_kda_fwd_kernel_inter_solve_fused - Triton: off-diagonal A_qk + A_kk solve
4. cutile_kernel.py    - cuTile: fused W, U, O, S update
"""

import torch
import torch.nn.functional as F
import triton

from fla.ops.utils.constant import RCP_LN2
# Import NEW optimized K2 kernel from FLA (safe_gate=True version)
from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_intra_sub_chunk
# Import K3 kernel (same as before)
from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused
from fla.utils import IS_GATHER_SUPPORTED

# Handle both package import and direct file import
try:
    from .cumsum_fused import chunk_local_cumsum_with_scaling
    from .cutile_kernel import launch_kda_kernel
except ImportError:
    from cumsum_fused import chunk_local_cumsum_with_scaling
    from cutile_kernel import launch_kda_kernel


def kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    normalize_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    KDA forward pass using optimized fused cuTile kernel.
    
    This implementation uses 4 kernels:
    1. Triton: Fused cumsum + scaling (parallel across chunks)
    2. Triton: A_qk, A_kkd computation (token-parallel)
    3. Triton: A_kk solve (chunk-parallel)
    4. cuTile: Fused W, U, O, S update (sequential across chunks)
    
    Key optimization: Beta is applied to A_kk in the cuTile kernel instead of
    precomputing V_scaled. This saves one [B, T, H, d] tensor.
    
    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Per-key gate tensor [B, T, H, K] (log-space, will be cumsum'd)
        beta: Beta tensor [B, T, H]
        scale: Attention scale (default: K^{-0.5})
        initial_state: Initial state [B, H, K, V] (default: zeros)
        output_final_state: Whether to return final state
        chunk_size: Chunk size for chunked attention (default: 64)
        normalize_qk: Whether to L2-normalize Q and K (default: True)
    
    Returns:
        output: Output tensor [B, T, H, V]
        final_state: Final state [B, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    device = q.device
    
    assert T % chunk_size == 0, f"seq_len ({T}) must be divisible by chunk_size ({chunk_size})"
    
    if scale is None:
        scale = K ** -0.5
    
    # L2 normalize Q, K if requested
    if normalize_qk:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
    
    # Initialize state
    if initial_state is None:
        h0 = torch.zeros(B, H, K, V_dim, dtype=torch.float32, device=device)
    else:
        h0 = initial_state.clone()
    
    # ========== Step 1: Fused cumsum + scaling (Triton, parallel) ==========
    # Computes: g_cumsum, k_scaled, kg, q_scaled, gk_last_exp
    # Note: V_scaled is NOT computed - saves one [B, T, H, d] tensor!
    g_cumsum, k_scaled, kg, q_scaled, gk_last_exp = chunk_local_cumsum_with_scaling(
        g=g,
        k=k,
        q=q,
        chunk_size=chunk_size,
        cumsum_scale=RCP_LN2,
        attn_scale=scale,
    )
    
    # ========== Step 2: A_qk, A_kkd computation (Triton, NEW optimized K2) ==========
    BT = chunk_size
    BC = 16
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    BK = triton.next_power_of_2(K)
    
    A_qk = torch.empty(B, T, H, BT, device=device, dtype=k.dtype)
    A_kk = torch.zeros(B, T, H, BT, device=device, dtype=k.dtype)
    A_kkd = torch.empty(B, T, H, BC, device=device, dtype=torch.float32)
    
    # NEW optimized K2: chunk_kda_fwd_kernel_intra_sub_chunk (safe_gate=True version)
    # This kernel computes diagonal blocks + does forward substitution internally
    grid_k2 = (NT, NC, B * H)
    chunk_kda_fwd_kernel_intra_sub_chunk[grid_k2](
        q=q, k=k, g=g_cumsum, beta=beta, Aqk=A_qk, Akk=A_kkd,
        scale=scale, cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    
    # ========== Step 3: A_kk solve (Triton, K3 with USE_SAFE_GATE=True) ==========
    grid_k3 = (NT, B * H)
    chunk_kda_fwd_kernel_inter_solve_fused[grid_k3](
        q=q, k=k, g=g_cumsum, beta=beta, Aqk=A_qk, Akkd=A_kkd, Akk=A_kk,
        scale=scale, cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=True,
    )
    
    # ========== Step 4: Fused kernel (cuTile, sequential across chunks) ==========
    num_chunks = T // chunk_size
    
    # Reshape tensors for kernel
    A_qk_r = A_qk.reshape(B, num_chunks, chunk_size, H, chunk_size)
    A_kk_r = A_kk.reshape(B, num_chunks, chunk_size, H, chunk_size)
    K_scaled_r = k_scaled.reshape(B, num_chunks, chunk_size, H, K)
    V_r = v.reshape(B, num_chunks, chunk_size, H, V_dim)  # Original V, not scaled
    Kg_r = kg.reshape(B, num_chunks, chunk_size, H, K)
    Q_scaled_r = q_scaled.reshape(B, num_chunks, chunk_size, H, K)
    Beta_r = beta.reshape(B, num_chunks, chunk_size, H)  # Beta for A_kk scaling
    
    # cuTile stores S as h^T
    S = h0.transpose(-1, -2).contiguous()
    
    # Launch kernel (beta applied to A_kk inside kernel)
    O_r = launch_kda_kernel(
        S=S,
        A_qk=A_qk_r,
        A_kk=A_kk_r,
        K_scaled=K_scaled_r,
        V=V_r,
        Kg=Kg_r,
        Q_scaled=Q_scaled_r,
        Beta=Beta_r,
        gk_last_exp=gk_last_exp,
        chunk_size=chunk_size
    )
    
    # Reshape output
    O = O_r.reshape(B, T, H, V_dim)
    
    if output_final_state:
        final_state = S.transpose(-1, -2).contiguous()
        return O, final_state
    
    return O, None
