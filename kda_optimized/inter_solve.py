# SPDX-FileCopyrightText: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# SPDX-License-Identifier: Apache-2.0

"""
Triton Kernel 3: Inter-chunk solve kernel for KDA.

Computes off-diagonal blocks of A_qk and solves the triangular system for A_kk.

This kernel is imported from FLA (flash-linear-attention) for compatibility.
"""

# Re-export from FLA
from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused

__all__ = ['chunk_kda_fwd_kernel_inter_solve_fused']
