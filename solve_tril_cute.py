# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Lower Triangular Matrix Inversion using CuTe DSL with TMA.

This module implements a 64x64 lower triangular matrix inverse kernel using CuTe Python DSL.
The algorithm follows the block-wise inversion approach from the Triton reference implementation.
Uses TMA (Tensor Memory Accelerator) for efficient global memory load/store.

Algorithm Overview:
    Given a 64x64 lower triangular matrix A (unit diagonal: I + StrictLower(A)):
    1. TMA Load: Load 64x64 block from global memory to shared memory
    2. Phase 1: Invert 4 diagonal 16x16 blocks in parallel (4 warps)
    3. Phase 2: Compute first-level off-diagonal blocks: Ai_21, Ai_32, Ai_43
    4. Phase 3: Compute second-level off-diagonal blocks: Ai_31, Ai_42
    5. Phase 4: Compute third-level off-diagonal block: Ai_41
    6. TMA Store: Store 64x64 result from shared memory to global memory

Usage:
    python solve_tril_cute.py

Reference:
    Triton implementation: solve_tril_triton.py
"""

import cutlass
import cutlass.cute as cute
from cutlass import utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op


# ===========================================================================
# Kernel Constants
# ===========================================================================
THREADS_PER_CTA = 128  # 4 warps
BLOCK_SIZE = 64
SUB_BLOCK_SIZE = 16


# ===========================================================================
# Device Kernel: Lower Triangular Matrix Inversion (64x64)
# ===========================================================================
@cute.kernel
def solve_tril_kernel(
    tma_load_atom: cute.CopyAtom,
    tma_store_atom: cute.CopyAtom,
    tma_load_tensor: cute.Tensor,
    tma_store_tensor: cute.Tensor,
):
    """
    64x64 Lower Triangular Matrix Inversion Kernel.
    
    Algorithm (Block Matrix Inversion):
    For matrix A = [A11  0    0    0  ]
                   [A21 A22   0    0  ]
                   [A31 A32  A33   0  ]
                   [A41 A42  A43  A44 ]
    
    Stage 1 (Diagonal Inversion): 4 warps in parallel
        - Warp 0: Ai11 = inv(A11)
        - Warp 1: Ai22 = inv(A22)
        - Warp 2: Ai33 = inv(A33)
        - Warp 3: Ai44 = inv(A44)
    
    Stage 2 (Off-diagonal Blocks):
        Phase 2a: Ai21 = -Ai22 * A21 * Ai11
                  Ai32 = -Ai33 * A32 * Ai22
                  Ai43 = -Ai44 * A43 * Ai33
        Phase 2b: Ai31 = -Ai33 * (A31 * Ai11 + A32 * Ai21)
                  Ai42 = -Ai44 * (A42 * Ai22 + A43 * Ai32)
        Phase 2c: Ai41 = -Ai44 * (A41 * Ai11 + A42 * Ai21 + A43 * Ai31)
    """
    # Thread indices
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32
    
    # Block indices - bidx is batch index
    batch_idx, _, _ = cute.arch.block_idx()
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    
    # SMEM layout for (1, 64, 64) float16 - 3D to match global tensor for TMA
    smem_layout_fp16 = cute.make_layout((1, BLOCK_SIZE, BLOCK_SIZE), stride=(0, BLOCK_SIZE, 1))
    sA_fp16 = smem.allocate_tensor(cutlass.Float16, smem_layout_fp16, 128)
    
    # SMEM layout for (4, 16, 16) float32 - only diagonal blocks
    # sA_fp32[block_idx, row, col] where block_idx = 0,1,2,3 for Ai11, Ai22, Ai33, Ai44
    smem_layout_fp32 = cute.make_layout((4, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE), stride=(SUB_BLOCK_SIZE * SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, 1))
    sA_fp32 = smem.allocate_tensor(cutlass.Float32, smem_layout_fp32, 128)
    
    # Allocate mbarrier for TMA
    mbar_ptr = smem.allocate_array(cutlass.Int64, 1)
    
    # Size of TMA transfer in bytes (64x64 float16 = 8KB)
    tile_bytes = BLOCK_SIZE * BLOCK_SIZE * 2
    
    # Initialize mbarrier (only thread 0)
    if tidx == 0:
        cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    
    # Get global tile for this CTA
    gA = cute.local_tile(tma_load_tensor, (1, BLOCK_SIZE, BLOCK_SIZE), (batch_idx, 0, None))
    gAi = cute.local_tile(tma_store_tensor, (1, BLOCK_SIZE, BLOCK_SIZE), (batch_idx, 0, None))
    
    gA_tile = gA[(0, None, None, 0)]
    gAi_tile = gAi[(0, None, None, 0)]
    sA_fp16_tile = sA_fp16[(0, None, None)]
    
    # =========================================================================
    # TMA Load: float16 Global -> SMEM
    # =========================================================================
    if warp_idx == 0:
        if tidx == 0:
            cute.arch.mbarrier_expect_tx(mbar_ptr, tile_bytes)
        
        tma_sDst, tma_gSrc = cpasync.tma_partition(
            tma_load_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA_fp16_tile, 0, 2),
            cute.group_modes(gA_tile, 0, 2),
        )
        cute.copy(tma_load_atom, tma_gSrc, tma_sDst, tma_bar_ptr=mbar_ptr)
    
    if warp_idx == 0 and tidx == 0:
        cute.arch.mbarrier_arrive(mbar_ptr)
    cute.arch.mbarrier_wait(mbar_ptr, 0)
    
    # =========================================================================
    # STAGE 1: Invert Diagonal 16x16 Blocks (4 half-warps in parallel)
    # =========================================================================
    # Read directly from sA_fp16, write to sA_fp32 diagonal buffer
    # Each warp splits into 2 half-warps of 16 threads
    # Half-warp 0 (warp 0, lanes 0-15): A11 at (0, 0) -> sA_fp32[0]
    # Half-warp 1 (warp 0, lanes 16-31): A22 at (16, 16) -> sA_fp32[1]
    # Half-warp 2 (warp 1, lanes 0-15): A33 at (32, 32) -> sA_fp32[2]
    # Half-warp 3 (warp 1, lanes 16-31): A44 at (48, 48) -> sA_fp32[3]
    # Only need 2 warps (64 threads) for 4 blocks
    
    # Wrap halfwarp_idx to 0-3 range (warps 2-3 redundantly compute same blocks)
    halfwarp_idx = (warp_idx * 2 + (lane_id // 16)) % 4  # 0, 1, 2, 3
    diag_offset = halfwarp_idx * SUB_BLOCK_SIZE
    
    # Each 16 threads handles one 16x16 block
    # Read from sA_fp16 at (diag_offset, diag_offset), write to sA_fp32[halfwarp_idx]
    _invert_16x16_halfwarp_impl(sA_fp16_tile, sA_fp32, diag_offset, halfwarp_idx, lane_id)
    cute.arch.sync_threads()
    
    # =========================================================================
    # STAGE 2: 6 个非对角块的第一乘 (tril @ dense)
    # =========================================================================
    # 6 个非对角块: Ai21, Ai31, Ai32, Ai41, Ai42, Ai43
    # 第一乘 (Ai_ii @ A_ij):
    #   0: Ai22 @ A21 -> [16:32,  0:16]
    #   1: Ai33 @ A31 -> [32:48,  0:16]
    #   2: Ai33 @ A32 -> [32:48, 16:32]
    #   3: Ai44 @ A41 -> [48:64,  0:16]
    #   4: Ai44 @ A42 -> [48:64, 16:32]
    #   5: Ai44 @ A43 -> [48:64, 32:48]
    #
    # 用 6 个 half-warp (3 个 warp), 每个 half-warp 16 线程算一个
    
    lane_id_16 = lane_id % 16
    halfwarp_idx = warp_idx * 2 + (lane_id // 16)  # 0-7
    
    if halfwarp_idx < 6:
        # 映射 halfwarp_idx 到 (tril_block, row_block, col_block)
        # halfwarp 0: Ai22 @ A21, tril=1, A21在[16:32, 0:16],  row=1, col=0
        # halfwarp 1: Ai33 @ A31, tril=2, A31在[32:48, 0:16],  row=2, col=0
        # halfwarp 2: Ai33 @ A32, tril=2, A32在[32:48, 16:32], row=2, col=1
        # halfwarp 3: Ai44 @ A41, tril=3, A41在[48:64, 0:16],  row=3, col=0
        # halfwarp 4: Ai44 @ A42, tril=3, A42在[48:64, 16:32], row=3, col=1
        # halfwarp 5: Ai44 @ A43, tril=3, A43在[48:64, 32:48], row=3, col=2
        
        # row_block: 1,2,2,3,3,3
        row_block = cutlass.Int32(halfwarp_idx == 0) * 1 + \
                    cutlass.Int32(halfwarp_idx >= 1) * cutlass.Int32(halfwarp_idx <= 2) * 2 + \
                    cutlass.Int32(halfwarp_idx >= 3) * 3
        
        # col_block: 0,0,1,0,1,2
        col_block = cutlass.Int32(halfwarp_idx == 2) * 1 + \
                    cutlass.Int32(halfwarp_idx == 4) * 1 + \
                    cutlass.Int32(halfwarp_idx == 5) * 2
        
        # tril_block = row_block (Ai22=1, Ai33=2, Ai44=3)
        tril_block = row_block
        
        # B (dense) 在 sA_fp16 的位置
        b_row_off = row_block * 16      # 16, 32, 32, 48, 48, 48
        b_col_off = col_block * 16      # 0, 0, 16, 0, 16, 32
        
        # 结果存回同样位置
        c_row_off = b_row_off
        c_col_off = b_col_off
        
        # tril2_block = col_block (Ai11=0, Ai22=1, Ai33=2)
        tril2_block = col_block
        
        _gemm_tril_a_dense_b_16t(
            sA_fp32,
            sA_fp16_tile,
            tril_block,
            b_row_off, b_col_off,
            c_row_off, c_col_off,
            lane_id_16,
            lane_id,
            tril2_block,
        )
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # Convert fp32 -> fp16 for diagonal blocks only
    # =========================================================================
    # Each warp handles one diagonal block: warp i -> sA_fp32[i] -> sA_fp16[diag_i]
    _warp_store_diag_fp32_to_fp16_smem(sA_fp32, sA_fp16_tile, warp_idx, lane_id)
    cute.arch.sync_threads()
    
    # =========================================================================
    # TMA Store: float16 SMEM -> Global
    # =========================================================================
    tma_sSrc, tma_gDst = cpasync.tma_partition(
        tma_store_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sA_fp16_tile, 0, 2),
        cute.group_modes(gAi_tile, 0, 2),
    )
    cute.copy(tma_store_atom, tma_sSrc, tma_gDst)


# ===========================================================================
# Helper: Store diagonal blocks from fp32 SMEM to fp16 SMEM
# ===========================================================================
@dsl_user_op
def _warp_store_diag_fp32_to_fp16_smem(
    sA_fp32: cute.Tensor,   # (4, 16, 16) fp32 - diagonal inverses
    sA_fp16: cute.Tensor,   # (64, 64) fp16 - output
    warp_idx,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    """
    Each warp copies one diagonal block from sA_fp32 to sA_fp16.
    Warp 0: sA_fp32[0] -> sA_fp16[0:16, 0:16]
    Warp 1: sA_fp32[1] -> sA_fp16[16:32, 16:32]
    Warp 2: sA_fp32[2] -> sA_fp16[32:48, 32:48]
    Warp 3: sA_fp32[3] -> sA_fp16[48:64, 48:64]
    
    Each warp has 32 threads, 16x16 = 256 elements, 8 elements per thread.
    """
    diag_offset = warp_idx * 16
    
    # Each thread handles 8 elements (256 / 32 = 8)
    for i in range(8):
        linear_idx = lane_id + i * 32
        local_row = linear_idx // 16
        local_col = linear_idx % 16
        
        # Load fp32 from diagonal buffer, convert to fp16, store to sA_fp16
        val_fp32 = sA_fp32[warp_idx, local_row, local_col]
        val_fp16 = cutlass.Float16(val_fp32)
        sA_fp16[diag_offset + local_row, diag_offset + local_col] = val_fp16


# ===========================================================================
# Helper Functions
# ===========================================================================

@dsl_user_op
def _invert_16x16_halfwarp_impl(
    sA_fp16: cute.Tensor,  # (64, 64) fp16 - source (original A)
    sA_fp32: cute.Tensor,  # (4, 16, 16) fp32 - destination (diagonal inverses)
    diag_offset,           # Offset in sA_fp16 (0, 16, 32, or 48)
    block_idx,             # Index in sA_fp32 (0, 1, 2, or 3)
    lane_id,               # 0-31 within the warp
    *,
    loc=None,
    ip=None,
):
    """
    Invert a 16x16 lower triangular block using 16 threads (half-warp).
    
    Reads from sA_fp16 at (diag_offset, diag_offset), writes to sA_fp32[block_idx].
    """
    my_row = lane_id % 16  # 0-15
    halfwarp_base = (lane_id // 16) * 16  # 0 for first half, 16 for second half
    
    row_off = diag_offset
    col_off = diag_offset
    
    # Diagonal-indexed registers
    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    rA = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    
    # Initialize
    for x in range(16):
        rInv[x] = cutlass.Float32(0.0)
        rA[x] = cutlass.Float32(0.0)
    
    # =========================================================================
    # Phase 1: Load from fp16 and compute by anti-diagonal
    # =========================================================================
    
    # d=1
    col1 = my_row - 1
    valid1 = cutlass.Float32(col1 >= 0)
    a_val1 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col1]) * valid1
    rA[1] = a_val1
    rInv[1] = -a_val1
    
    # d=2
    col2 = my_row - 2
    valid2 = cutlass.Float32(col2 >= 0)
    a_val2 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col2]) * valid2
    rA[2] = a_val2
    inv_from_row1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 1)
    sum2 = rA[1] * inv_from_row1
    rInv[2] = (-a_val2 - sum2) * valid2
    
    # d=3
    col3 = my_row - 3
    valid3 = cutlass.Float32(col3 >= 0)
    a_val3 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col3]) * valid3
    rA[3] = a_val3
    inv_from_row2_d1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 2)
    inv_from_row1_d2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 1)
    sum3 = rA[2] * inv_from_row2_d1 + rA[1] * inv_from_row1_d2
    rInv[3] = (-a_val3 - sum3) * valid3
    
    # d=4
    col4 = my_row - 4
    valid4 = cutlass.Float32(col4 >= 0)
    a_val4 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col4]) * valid4
    rA[4] = a_val4
    inv_d4_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 3)
    inv_d4_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 2)
    inv_d4_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 1)
    sum4 = rA[3] * inv_d4_k1 + rA[2] * inv_d4_k2 + rA[1] * inv_d4_k3
    rInv[4] = (-a_val4 - sum4) * valid4
    
    # d=5
    col5 = my_row - 5
    valid5 = cutlass.Float32(col5 >= 0)
    a_val5 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col5]) * valid5
    rA[5] = a_val5
    inv_d5_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 4)
    inv_d5_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 3)
    inv_d5_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 2)
    inv_d5_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 1)
    sum5 = rA[4] * inv_d5_k1 + rA[3] * inv_d5_k2 + rA[2] * inv_d5_k3 + rA[1] * inv_d5_k4
    rInv[5] = (-a_val5 - sum5) * valid5
    
    # d=6
    col6 = my_row - 6
    valid6 = cutlass.Float32(col6 >= 0)
    a_val6 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col6]) * valid6
    rA[6] = a_val6
    inv_d6_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 5)
    inv_d6_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 4)
    inv_d6_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 3)
    inv_d6_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 2)
    inv_d6_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 1)
    sum6 = rA[5] * inv_d6_k1 + rA[4] * inv_d6_k2 + rA[3] * inv_d6_k3 + rA[2] * inv_d6_k4 + rA[1] * inv_d6_k5
    rInv[6] = (-a_val6 - sum6) * valid6
    
    # d=7
    col7 = my_row - 7
    valid7 = cutlass.Float32(col7 >= 0)
    a_val7 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col7]) * valid7
    rA[7] = a_val7
    inv_d7_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 6)
    inv_d7_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 5)
    inv_d7_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 4)
    inv_d7_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 3)
    inv_d7_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 2)
    inv_d7_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 1)
    sum7 = rA[6] * inv_d7_k1 + rA[5] * inv_d7_k2 + rA[4] * inv_d7_k3 + rA[3] * inv_d7_k4 + rA[2] * inv_d7_k5 + rA[1] * inv_d7_k6
    rInv[7] = (-a_val7 - sum7) * valid7
    
    # d=8
    col8 = my_row - 8
    valid8 = cutlass.Float32(col8 >= 0)
    a_val8 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col8]) * valid8
    rA[8] = a_val8
    inv_d8_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 7)
    inv_d8_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 6)
    inv_d8_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 5)
    inv_d8_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 4)
    inv_d8_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 3)
    inv_d8_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 2)
    inv_d8_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 1)
    sum8 = rA[7] * inv_d8_k1 + rA[6] * inv_d8_k2 + rA[5] * inv_d8_k3 + rA[4] * inv_d8_k4 + rA[3] * inv_d8_k5 + rA[2] * inv_d8_k6 + rA[1] * inv_d8_k7
    rInv[8] = (-a_val8 - sum8) * valid8
    
    # d=9
    col9 = my_row - 9
    valid9 = cutlass.Float32(col9 >= 0)
    a_val9 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col9]) * valid9
    rA[9] = a_val9
    inv_d9_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 8)
    inv_d9_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 7)
    inv_d9_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 6)
    inv_d9_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 5)
    inv_d9_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 4)
    inv_d9_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 3)
    inv_d9_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 2)
    inv_d9_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 1)
    sum9 = rA[8] * inv_d9_k1 + rA[7] * inv_d9_k2 + rA[6] * inv_d9_k3 + rA[5] * inv_d9_k4 + rA[4] * inv_d9_k5 + rA[3] * inv_d9_k6 + rA[2] * inv_d9_k7 + rA[1] * inv_d9_k8
    rInv[9] = (-a_val9 - sum9) * valid9
    
    # d=10
    col10 = my_row - 10
    valid10 = cutlass.Float32(col10 >= 0)
    a_val10 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col10]) * valid10
    rA[10] = a_val10
    inv_d10_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 9)
    inv_d10_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 8)
    inv_d10_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 7)
    inv_d10_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 6)
    inv_d10_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 5)
    inv_d10_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 4)
    inv_d10_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 3)
    inv_d10_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 2)
    inv_d10_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 1)
    sum10 = rA[9] * inv_d10_k1 + rA[8] * inv_d10_k2 + rA[7] * inv_d10_k3 + rA[6] * inv_d10_k4 + rA[5] * inv_d10_k5 + rA[4] * inv_d10_k6 + rA[3] * inv_d10_k7 + rA[2] * inv_d10_k8 + rA[1] * inv_d10_k9
    rInv[10] = (-a_val10 - sum10) * valid10
    
    # d=11
    col11 = my_row - 11
    valid11 = cutlass.Float32(col11 >= 0)
    a_val11 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col11]) * valid11
    rA[11] = a_val11
    inv_d11_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 10)
    inv_d11_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 9)
    inv_d11_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 8)
    inv_d11_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 7)
    inv_d11_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 6)
    inv_d11_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 5)
    inv_d11_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 4)
    inv_d11_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 3)
    inv_d11_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 2)
    inv_d11_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 1)
    sum11 = rA[10] * inv_d11_k1 + rA[9] * inv_d11_k2 + rA[8] * inv_d11_k3 + rA[7] * inv_d11_k4 + rA[6] * inv_d11_k5 + rA[5] * inv_d11_k6 + rA[4] * inv_d11_k7 + rA[3] * inv_d11_k8 + rA[2] * inv_d11_k9 + rA[1] * inv_d11_k10
    rInv[11] = (-a_val11 - sum11) * valid11
    
    # d=12
    col12 = my_row - 12
    valid12 = cutlass.Float32(col12 >= 0)
    a_val12 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col12]) * valid12
    rA[12] = a_val12
    inv_d12_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 11)
    inv_d12_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 10)
    inv_d12_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 9)
    inv_d12_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 8)
    inv_d12_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 7)
    inv_d12_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 6)
    inv_d12_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 5)
    inv_d12_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 4)
    inv_d12_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 3)
    inv_d12_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 2)
    inv_d12_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 1)
    sum12 = rA[11] * inv_d12_k1 + rA[10] * inv_d12_k2 + rA[9] * inv_d12_k3 + rA[8] * inv_d12_k4 + rA[7] * inv_d12_k5 + rA[6] * inv_d12_k6 + rA[5] * inv_d12_k7 + rA[4] * inv_d12_k8 + rA[3] * inv_d12_k9 + rA[2] * inv_d12_k10 + rA[1] * inv_d12_k11
    rInv[12] = (-a_val12 - sum12) * valid12
    
    # d=13
    col13 = my_row - 13
    valid13 = cutlass.Float32(col13 >= 0)
    a_val13 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col13]) * valid13
    rA[13] = a_val13
    inv_d13_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 12)
    inv_d13_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 11)
    inv_d13_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 10)
    inv_d13_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 9)
    inv_d13_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 8)
    inv_d13_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 7)
    inv_d13_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 6)
    inv_d13_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 5)
    inv_d13_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 4)
    inv_d13_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 3)
    inv_d13_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 2)
    inv_d13_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 1)
    sum13 = rA[12] * inv_d13_k1 + rA[11] * inv_d13_k2 + rA[10] * inv_d13_k3 + rA[9] * inv_d13_k4 + rA[8] * inv_d13_k5 + rA[7] * inv_d13_k6 + rA[6] * inv_d13_k7 + rA[5] * inv_d13_k8 + rA[4] * inv_d13_k9 + rA[3] * inv_d13_k10 + rA[2] * inv_d13_k11 + rA[1] * inv_d13_k12
    rInv[13] = (-a_val13 - sum13) * valid13
    
    # d=14
    col14 = my_row - 14
    valid14 = cutlass.Float32(col14 >= 0)
    a_val14 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col14]) * valid14
    rA[14] = a_val14
    inv_d14_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 13)
    inv_d14_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 12)
    inv_d14_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 11)
    inv_d14_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 10)
    inv_d14_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 9)
    inv_d14_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 8)
    inv_d14_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 7)
    inv_d14_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 6)
    inv_d14_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 5)
    inv_d14_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 4)
    inv_d14_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 3)
    inv_d14_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 2)
    inv_d14_k13 = cute.arch.shuffle_sync(rInv[13], halfwarp_base + my_row - 1)
    sum14 = rA[13] * inv_d14_k1 + rA[12] * inv_d14_k2 + rA[11] * inv_d14_k3 + rA[10] * inv_d14_k4 + rA[9] * inv_d14_k5 + rA[8] * inv_d14_k6 + rA[7] * inv_d14_k7 + rA[6] * inv_d14_k8 + rA[5] * inv_d14_k9 + rA[4] * inv_d14_k10 + rA[3] * inv_d14_k11 + rA[2] * inv_d14_k12 + rA[1] * inv_d14_k13
    rInv[14] = (-a_val14 - sum14) * valid14
    
    # d=15
    col15 = my_row - 15
    valid15 = cutlass.Float32(col15 >= 0)
    a_val15 = cutlass.Float32(sA_fp16[row_off + my_row, col_off + col15]) * valid15
    rA[15] = a_val15
    inv_d15_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 14)
    inv_d15_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 13)
    inv_d15_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 12)
    inv_d15_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 11)
    inv_d15_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 10)
    inv_d15_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 9)
    inv_d15_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 8)
    inv_d15_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 7)
    inv_d15_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 6)
    inv_d15_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 5)
    inv_d15_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 4)
    inv_d15_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 3)
    inv_d15_k13 = cute.arch.shuffle_sync(rInv[13], halfwarp_base + my_row - 2)
    inv_d15_k14 = cute.arch.shuffle_sync(rInv[14], halfwarp_base + my_row - 1)
    sum15 = rA[14] * inv_d15_k1 + rA[13] * inv_d15_k2 + rA[12] * inv_d15_k3 + rA[11] * inv_d15_k4 + rA[10] * inv_d15_k5 + rA[9] * inv_d15_k6 + rA[8] * inv_d15_k7 + rA[7] * inv_d15_k8 + rA[6] * inv_d15_k9 + rA[5] * inv_d15_k10 + rA[4] * inv_d15_k11 + rA[3] * inv_d15_k12 + rA[2] * inv_d15_k13 + rA[1] * inv_d15_k14
    rInv[15] = (-a_val15 - sum15) * valid15
    
    # Set diagonal to 1
    rInv[0] = cutlass.Float32(1.0)
    
    # =========================================================================
    # Phase 2: Store to sA_fp32[block_idx] diagonal buffer
    # =========================================================================
    # Store diagonal first
    sA_fp32[block_idx, my_row, my_row] = rInv[0]
    
    # Store off-diagonals (write 0 for upper triangular part)
    sA_fp32[block_idx, my_row, my_row - 1] = rInv[1] * cutlass.Float32(my_row >= 1)
    sA_fp32[block_idx, my_row, my_row - 2] = rInv[2] * cutlass.Float32(my_row >= 2)
    sA_fp32[block_idx, my_row, my_row - 3] = rInv[3] * cutlass.Float32(my_row >= 3)
    sA_fp32[block_idx, my_row, my_row - 4] = rInv[4] * cutlass.Float32(my_row >= 4)
    sA_fp32[block_idx, my_row, my_row - 5] = rInv[5] * cutlass.Float32(my_row >= 5)
    sA_fp32[block_idx, my_row, my_row - 6] = rInv[6] * cutlass.Float32(my_row >= 6)
    sA_fp32[block_idx, my_row, my_row - 7] = rInv[7] * cutlass.Float32(my_row >= 7)
    sA_fp32[block_idx, my_row, my_row - 8] = rInv[8] * cutlass.Float32(my_row >= 8)
    sA_fp32[block_idx, my_row, my_row - 9] = rInv[9] * cutlass.Float32(my_row >= 9)
    sA_fp32[block_idx, my_row, my_row - 10] = rInv[10] * cutlass.Float32(my_row >= 10)
    sA_fp32[block_idx, my_row, my_row - 11] = rInv[11] * cutlass.Float32(my_row >= 11)
    sA_fp32[block_idx, my_row, my_row - 12] = rInv[12] * cutlass.Float32(my_row >= 12)
    sA_fp32[block_idx, my_row, my_row - 13] = rInv[13] * cutlass.Float32(my_row >= 13)
    sA_fp32[block_idx, my_row, my_row - 14] = rInv[14] * cutlass.Float32(my_row >= 14)
    sA_fp32[block_idx, my_row, my_row - 15] = rInv[15] * cutlass.Float32(my_row >= 15)


@dsl_user_op
def _gemm_tril_a_dense_b_16t(
    sA_fp32: cute.Tensor,  # (4, 16, 16) fp32 - source for A (lower triangular inverse)
    sA_fp16: cute.Tensor,  # (64, 64) fp16 - source for B (dense) and dest for C
    a_block_idx,           # Index of A block in sA_fp32 (0, 1, 2, or 3)
    b_row_off,             # Row offset for B block in sA_fp16
    b_col_off,             # Col offset for B block in sA_fp16
    c_row_off,             # Row offset for C block in sA_fp16
    c_col_off,             # Col offset for C block in sA_fp16
    lane_id_16,            # 0-15 within the 16-thread group
    lane_id,               # 0-31 within warp (for shuffle)
    tril2_block,           # Index of tril block for second multiply
    *,
    loc=None,
    ip=None,
):
    """
    Compute Result = -(A @ B) @ Ai2 where:
      - A is lower triangular (16x16)
      - B is dense (16x16)
      - Ai2 is lower triangular (16x16)
    Uses 16 threads.
    
    Algorithm:
    1. Preload column j of B (dense) into registers
    2. Compute C = A @ B (each thread has one column)
    3. Shuffle transpose: each thread now has one row
    4. Compute -C @ Ai2 (dense @ tril)
    5. Write back row-wise as fp16
    """
    my_col = lane_id_16  # Thread j handles column j of C
    halfwarp_base = (lane_id // 16) * 16
    
    # =========================================================================
    # Step 1: Preload column my_col of B (dense)
    # =========================================================================
    rB = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    for k in range(16):
        rB[k] = cutlass.Float32(sA_fp16[b_row_off + k, b_col_off + my_col])
    
    # =========================================================================
    # Step 2: Compute C = A @ B (tril @ dense)
    # =========================================================================
    rC = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    
    for i in range(16):
        acc = cutlass.Float32(0.0)
        for k in range(i + 1):  # A is lower triangular
            a_ik = sA_fp32[a_block_idx, i, k]
            acc = acc + a_ik * rB[k]
        rC[i] = acc
    
    # =========================================================================
    # Step 3: Shuffle transpose (column-wise -> row-wise)
    # =========================================================================
    # Before: thread j has rC[i] = C[i, j] (column j)
    # After:  thread i has rC_T[k] = C[i, k] (row i)
    my_row = lane_id_16
    rC_T = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    
    for k in range(16):
        rC_T[k] = cute.arch.shuffle_sync(rC[my_row], halfwarp_base + k)
    
    # =========================================================================
    # Step 4: Second multiply: Result = -C @ Ai2 (dense @ tril)
    # =========================================================================
    # Thread i has row i of C in rC_T[k] = C[i, k]
    # Result[i, j] = -sum_{k>=j} C[i, k] * Ai2[k, j]
    # Ai2 is lower triangular: Ai2[k, j] = 0 for k < j
    
    rResult = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    for j in range(16):
        rResult[j] = cutlass.Float32(0.0)
    
    for k in range(16):
        c_ik = rC_T[k]
        # Row k of Ai2 has non-zero at columns 0..k
        for j in range(k + 1):
            ai2_kj = sA_fp32[tril2_block, k, j]
            rResult[j] = rResult[j] + c_ik * ai2_kj
    
    # =========================================================================
    # Step 5: Negate, shuffle transpose (row -> column), write as fp16
    # =========================================================================
    # Negate in place
    for j in range(16):
        rResult[j] = -rResult[j]
    
    # Shuffle transpose: row-wise -> column-wise (reuse rC_T)
    # Before: thread i has row i: rResult[j] = -Result[i, j]
    # After:  thread j has column j: rC_T[i] = -Result[i, j]
    my_col = lane_id_16
    
    for i in range(16):
        # -Result[i, my_col] is in thread i's rResult[my_col]
        rC_T[i] = cute.arch.shuffle_sync(rResult[my_col], halfwarp_base + i)
    
    # Write column my_col, convert to fp16
    for i in range(16):
        sA_fp16[c_row_off + i, c_col_off + my_col] = cutlass.Float16(rC_T[i])


# ===========================================================================
# Host JIT Function
# ===========================================================================
@cute.jit
def solve_tril_host(
    A: cute.Tensor,
    Ai: cute.Tensor,
    batch_size: cutlass.Constexpr[int],
):
    """
    Host function to set up TMA and launch the solve_tril kernel.
    
    :param A: Input tensor - 3D lower triangular matrices (batch, 64, 64)
    :param Ai: Output tensor - inverse matrices (same shape as A)
    :param batch_size: Number of matrices in the batch
    """
    # Define shared memory layout for 64x64 float64 (row-major)
    # For 3D tensor (batch, 64, 64), we tile (1, 64, 64)
    smem_layout = cute.make_layout((1, BLOCK_SIZE, BLOCK_SIZE), stride=(0, BLOCK_SIZE, 1))
    
    # CTA tiler - how much each CTA processes
    cta_tiler = cute.product_each(smem_layout.shape)
    
    # Create TMA load atom (Global -> Shared)
    tma_load_atom, tma_load_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        A,
        smem_layout,
        cta_tiler,
    )
    
    # Create TMA store atom (Shared -> Global)
    tma_store_atom, tma_store_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        Ai,
        smem_layout,
        cta_tiler,
    )
    
    # Compute grid dimensions
    # Each CTA processes one 64x64 matrix from the batch
    grid = (batch_size, 1, 1)
    
    # Launch kernel
    solve_tril_kernel(
        tma_load_atom,
        tma_store_atom,
        tma_load_tensor,
        tma_store_tensor,
    ).launch(
        grid=grid,
        block=(THREADS_PER_CTA, 1, 1),
    )


# ===========================================================================
# Test Function
# ===========================================================================
def test_solve_tril_cute():
    """
    Benchmark the solve_tril kernel with TMA load/store.
    Uses batch dimension and float64.
    """
    import torch
    import time
    
    # Test parameters
    BATCH_SIZE = 32 * 1024
    WARMUP_ITERS = 5
    BENCH_ITERS = 100
    
    print("=" * 60)
    print("Benchmarking Lower Triangular Matrix Inversion with TMA")
    print("=" * 60)
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Matrix size: 64x64")
    print(f"  Data type: float16")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Benchmark iterations: {BENCH_ITERS}")
    
    # Initialize CUDA context
    cutlass.cuda.initialize_cuda_context()
    
    # Create input tensor with shape (batch, 64, 64) in float16
    torch.manual_seed(42)
    A = torch.randn(BATCH_SIZE, 64, 64, device="cuda", dtype=torch.float16)
    A = A.tril(-1)  # Strictly lower triangular (diagonal = 0)
    
    # Output tensor
    Ai = torch.zeros_like(A)
    
    # Convert to cute tensors with proper alignment
    A_cute = from_dlpack(A, assumed_align=16)
    Ai_cute = from_dlpack(Ai, assumed_align=16)
    
    print(f"\n  Input tensor shape: {A.shape}")
    print(f"  Input tensor dtype: {A.dtype}")
    
    # Compile kernel once (like fla.py pattern)
    print("\nCompiling kernel...")
    compiled_kernel = cute.compile(
        solve_tril_host,
        A_cute,
        Ai_cute,
        BATCH_SIZE,
    )
    torch.cuda.synchronize()
    print("Compilation done.")
    
    # Warmup
    print(f"\nWarmup ({WARMUP_ITERS} iterations)...")
    for _ in range(WARMUP_ITERS):
        compiled_kernel(A_cute, Ai_cute)
    torch.cuda.synchronize()
    
    # Benchmark - measure total time for all iterations
    print(f"\nBenchmarking ({BENCH_ITERS} iterations)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(BENCH_ITERS):
        compiled_kernel(A_cute, Ai_cute)
    end_event.record()
    
    torch.cuda.synchronize()
    
    # Calculate timings
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / BENCH_ITERS
    
    # Calculate throughput
    data_size_mb = BATCH_SIZE * 64 * 64 * 2 / (1024 * 1024)  # float16 = 2 bytes
    total_data_mb = data_size_mb * 2  # Read + Write
    bandwidth_gbps = total_data_mb / (mean_time / 1000) / 1024
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Total time:    {total_time_ms:.3f} ms ({BENCH_ITERS} iterations)")
    print(f"  Time per iter: {mean_time:.3f} ms")
    print(f"  Data per iter: {data_size_mb:.2f} MB (read) + {data_size_mb:.2f} MB (write)")
    print(f"  Bandwidth:     {bandwidth_gbps:.1f} GB/s")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_solve_tril_cute()
    
    if success:
        print("\nBenchmark completed!")
    else:
        print("\nBenchmark failed!")
        exit(1)


