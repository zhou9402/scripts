# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
KDA Optimized - cuTile + Triton Fused Implementation

This package contains an optimized KDA (Key-Dependent Attention) implementation
that fuses multiple kernels for better performance.

Kernels:
1. cumsum_fused.py     - Triton: cumsum + all scaling operations
2. intra_parallel.py   - Triton: A_qk, A_kkd computation (token-parallel)
3. inter_solve.py      - Triton: A_kk solve (from FLA)
4. cutile_kernel.py    - cuTile: fused W, U, O, S update

Usage:
    from kda_optimized import kda_forward
    
    O, S = kda_forward(Q, K, V, G, Beta, scale, h0, chunk_size=64)
"""

# Handle both package import and direct file import
try:
    from .forward import kda_forward
except ImportError:
    from forward import kda_forward

__all__ = ['kda_forward']
