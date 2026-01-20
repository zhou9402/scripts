# SPDX-FileCopyrightText: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# SPDX-License-Identifier: Apache-2.0

"""
Triton Kernel 2: Token-parallel implementation of KDA intra chunk kernel.

Computes A_qk (query-key attention) and A_kkd (diagonal blocks of key-key attention).

This kernel is imported from FLA (flash-linear-attention) for compatibility.
"""

# Re-export from FLA
from fla.ops.kda.chunk_intra_token_parallel import (
    chunk_kda_fwd_kernel_intra_token_parallel,
    chunk_kda_fwd_intra_token_parallel,
)

__all__ = [
    'chunk_kda_fwd_kernel_intra_token_parallel',
    'chunk_kda_fwd_intra_token_parallel',
]
