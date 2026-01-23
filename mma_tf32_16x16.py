# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
TF32 MMA 16x16 Matrix Multiplication using CuTe DSL with inline PTX.

TF32 MMA instruction: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
- Shape: M=16, N=8, K=8
- To compute 16x16 @ 16x16 = 16x16, we need:
  - N direction: 16/8 = 2 MMA ops
  - K direction: 16/8 = 2 accumulations
  - Total: 4 MMA ops

Register layout per thread (32 threads in warp):
- A (16x8): 4 FP32 registers
- B (8x8): 2 FP32 registers  
- C/D (16x8): 4 FP32 registers
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


THREADS_PER_CTA = 32  # 1 warp


# ===========================================================================
# TF32 MMA Inline PTX Wrapper
# ===========================================================================
@dsl_user_op
def mma_tf32_m16n8k8(
    a0, a1, a2, a3,      # A matrix: 4 FP32 registers
    b0, b1,              # B matrix: 2 FP32 registers
    c0, c1, c2, c3,      # C accumulator: 4 FP32 registers (input)
    *, loc=None, ip=None
):
    """
    TF32 MMA: D = A * B + C
    Shape: m16n8k8, A is row-major, B is col-major
    
    Returns: (d0, d1, d2, d3) - 4 FP32 results
    
    Note: TF32 inputs are passed as f32 but the MMA instruction interprets
    them as tf32 (truncated to 19 bits of precision).
    """
    # Convert f32 to i32 for A and B operands (tf32 uses bit representation)
    # PTX expects tf32 operands in .b32 registers
    a0_bits = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1_bits = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2_bits = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3_bits = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0_bits = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1_bits = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    
    # Use inline_asm to emit mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    result = llvm.inline_asm(
        # Return type: 4 x f32
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        # Input operands: A (as i32), B (as i32), C (as f32)
        [
            a0_bits,
            a1_bits,
            a2_bits,
            a3_bits,
            b0_bits,
            b1_bits,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        # PTX assembly
        """{
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        # Constraints: 4 f32 outputs, 4 r32 A inputs, 2 r32 B inputs, 4 f32 C inputs
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    
    # Extract results
    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    
    return d0, d1, d2, d3


# ===========================================================================
# Load A matrix from SMEM to registers (TF32 MMA layout)
# ===========================================================================
@dsl_user_op
def load_A_tf32(
    sA: cute.Tensor,  # (16, 8) FP32 in SMEM, row-major
    lane_id,
    *, loc=None, ip=None
):
    """
    Load A matrix (16x8) from SMEM to registers for TF32 MMA.
    
    Register layout (from PTX ISA docs for mma.sync.m16n8k8.row.col):
    - group_id = lane_id / 4 (0-7)
    - tid_in_group = lane_id % 4 (0-3)
    - a0 = A[group_id,     tid_in_group]
    - a1 = A[group_id + 8, tid_in_group]      # NOT col + 4!
    - a2 = A[group_id,     tid_in_group + 4]  # NOT row + 8!
    - a3 = A[group_id + 8, tid_in_group + 4]
    """
    group_id = lane_id // 4
    tid_in_group = lane_id % 4
    
    a0 = cutlass.Float32(sA[group_id, tid_in_group])
    a1 = cutlass.Float32(sA[group_id + 8, tid_in_group])
    a2 = cutlass.Float32(sA[group_id, tid_in_group + 4])
    a3 = cutlass.Float32(sA[group_id + 8, tid_in_group + 4])
    
    return a0, a1, a2, a3


# ===========================================================================
# Load B matrix from SMEM to registers (TF32 MMA layout)
# ===========================================================================
@dsl_user_op
def load_B_tf32(
    sB: cute.Tensor,  # (8, 8) FP32 in SMEM, col-major (stride=(1, 8))
    lane_id,
    *, loc=None, ip=None
):
    """
    Load B matrix (8x8) from SMEM to registers for TF32 MMA.
    B is col-major in SMEM.
    
    Register layout:
    - group_id = lane_id / 4 (0-7, corresponds to column)
    - tid_in_group = lane_id % 4 (0-3, corresponds to row offset)
    - b0 = B[tid_in_group, group_id]
    - b1 = B[tid_in_group + 4, group_id]
    """
    group_id = lane_id // 4
    tid_in_group = lane_id % 4
    
    b0 = cutlass.Float32(sB[tid_in_group, group_id])
    b1 = cutlass.Float32(sB[tid_in_group + 4, group_id])
    
    return b0, b1


# ===========================================================================
# Load B matrix from SMEM (row-major) with transpose for TF32 MMA
# ===========================================================================
@dsl_user_op
def load_B_tf32_from_rowmajor(
    sB: cute.Tensor,  # (8, 8) FP32 in SMEM, row-major (stride=(8, 1))
    lane_id,
    *, loc=None, ip=None
):
    """
    Load B matrix (8x8) from row-major SMEM to registers for TF32 MMA.
    MMA expects B in col-major, so we load transposed.
    
    For B in row-major: B[row, col] at address row*stride + col
    MMA wants col-major view, so we read:
    - b0 = B[group_id, tid_in_group] (treating as if col-major)
    - b1 = B[group_id, tid_in_group + 4]
    
    This effectively transposes B during load.
    """
    group_id = lane_id // 4
    tid_in_group = lane_id % 4
    
    # Load with transposed indices (swap row/col)
    b0 = cutlass.Float32(sB[tid_in_group, group_id])
    b1 = cutlass.Float32(sB[tid_in_group + 4, group_id])
    
    return b0, b1


# ===========================================================================
# Store C/D matrix from registers to SMEM
# ===========================================================================
@dsl_user_op
def store_C_tf32(
    sC: cute.Tensor,  # (16, 8) FP32 in SMEM, row-major
    c0, c1, c2, c3,   # 4 FP32 registers
    lane_id,
    *, loc=None, ip=None
):
    """
    Store C/D matrix (16x8) from registers to SMEM.
    
    Register layout:
    - group_id = lane_id / 4 (0-7)
    - tid_in_group = lane_id % 4 (0-3)
    - c0 -> C[group_id, tid_in_group * 2]
    - c1 -> C[group_id, tid_in_group * 2 + 1]
    - c2 -> C[group_id + 8, tid_in_group * 2]
    - c3 -> C[group_id + 8, tid_in_group * 2 + 1]
    """
    group_id = lane_id // 4
    tid_in_group = lane_id % 4
    
    sC[group_id, tid_in_group * 2] = c0
    sC[group_id, tid_in_group * 2 + 1] = c1
    sC[group_id + 8, tid_in_group * 2] = c2
    sC[group_id + 8, tid_in_group * 2 + 1] = c3


# ===========================================================================
# 16x16 @ 16x16 Matrix Multiplication using TF32 MMA
# ===========================================================================
@dsl_user_op
def gemm_tf32_16x16(
    sA: cute.Tensor,   # (16, 16) FP32 in SMEM, row-major
    sB: cute.Tensor,   # (16, 16) FP32 in SMEM, row-major
    sC: cute.Tensor,   # (16, 16) FP32 in SMEM, row-major (output)
    lane_id,
    *, loc=None, ip=None
):
    """
    Compute C = A @ B where A, B, C are 16x16 FP32 matrices.
    Uses 4 TF32 MMA operations (m16n8k8).
    
    Decomposition:
    - C[:, 0:8]  = A[:, 0:8] @ B[0:8, 0:8] + A[:, 8:16] @ B[8:16, 0:8]
    - C[:, 8:16] = A[:, 0:8] @ B[0:8, 8:16] + A[:, 8:16] @ B[8:16, 8:16]
    
    That's 4 MMA ops total.
    """
    # Initialize accumulators for C0 (16x8) and C1 (16x8)
    c0_0 = cutlass.Float32(0.0)
    c0_1 = cutlass.Float32(0.0)
    c0_2 = cutlass.Float32(0.0)
    c0_3 = cutlass.Float32(0.0)
    
    c1_0 = cutlass.Float32(0.0)
    c1_1 = cutlass.Float32(0.0)
    c1_2 = cutlass.Float32(0.0)
    c1_3 = cutlass.Float32(0.0)
    
    # =========================================================================
    # K iteration 0: K = 0..7
    # =========================================================================
    # Load A[:, 0:8] (16x8)
    sA_k0 = cute.local_tile(sA, tiler=(16, 8), coord=(0, 0))
    a0_k0, a1_k0, a2_k0, a3_k0 = load_A_tf32(sA_k0, lane_id)
    
    # Load B[0:8, 0:8] for C0 (need col-major view, but SMEM is row-major)
    # B block at (0, 0)
    sB_00 = cute.local_tile(sB, tiler=(8, 8), coord=(0, 0))
    b0_00, b1_00 = load_B_tf32_from_rowmajor(sB_00, lane_id)
    
    # MMA #1: C0 += A[:, 0:8] @ B[0:8, 0:8]
    c0_0, c0_1, c0_2, c0_3 = mma_tf32_m16n8k8(
        a0_k0, a1_k0, a2_k0, a3_k0,
        b0_00, b1_00,
        c0_0, c0_1, c0_2, c0_3
    )
    
    # Load B[0:8, 8:16] for C1
    sB_01 = cute.local_tile(sB, tiler=(8, 8), coord=(0, 1))
    b0_01, b1_01 = load_B_tf32_from_rowmajor(sB_01, lane_id)
    
    # MMA #2: C1 += A[:, 0:8] @ B[0:8, 8:16]
    c1_0, c1_1, c1_2, c1_3 = mma_tf32_m16n8k8(
        a0_k0, a1_k0, a2_k0, a3_k0,
        b0_01, b1_01,
        c1_0, c1_1, c1_2, c1_3
    )
    
    # =========================================================================
    # K iteration 1: K = 8..15
    # =========================================================================
    # Load A[:, 8:16] (16x8)
    sA_k1 = cute.local_tile(sA, tiler=(16, 8), coord=(0, 1))
    a0_k1, a1_k1, a2_k1, a3_k1 = load_A_tf32(sA_k1, lane_id)
    
    # Load B[8:16, 0:8] for C0
    sB_10 = cute.local_tile(sB, tiler=(8, 8), coord=(1, 0))
    b0_10, b1_10 = load_B_tf32_from_rowmajor(sB_10, lane_id)
    
    # MMA #3: C0 += A[:, 8:16] @ B[8:16, 0:8]
    c0_0, c0_1, c0_2, c0_3 = mma_tf32_m16n8k8(
        a0_k1, a1_k1, a2_k1, a3_k1,
        b0_10, b1_10,
        c0_0, c0_1, c0_2, c0_3
    )
    
    # Load B[8:16, 8:16] for C1
    sB_11 = cute.local_tile(sB, tiler=(8, 8), coord=(1, 1))
    b0_11, b1_11 = load_B_tf32_from_rowmajor(sB_11, lane_id)
    
    # MMA #4: C1 += A[:, 8:16] @ B[8:16, 8:16]
    c1_0, c1_1, c1_2, c1_3 = mma_tf32_m16n8k8(
        a0_k1, a1_k1, a2_k1, a3_k1,
        b0_11, b1_11,
        c1_0, c1_1, c1_2, c1_3
    )
    
    # =========================================================================
    # Store results
    # =========================================================================
    # Store C0 to C[:, 0:8]
    sC_0 = cute.local_tile(sC, tiler=(16, 8), coord=(0, 0))
    store_C_tf32(sC_0, c0_0, c0_1, c0_2, c0_3, lane_id)
    
    # Store C1 to C[:, 8:16]
    sC_1 = cute.local_tile(sC, tiler=(16, 8), coord=(0, 1))
    store_C_tf32(sC_1, c1_0, c1_1, c1_2, c1_3, lane_id)


# ===========================================================================
# Test Kernel
# ===========================================================================
@cute.kernel
def test_tf32_gemm_kernel(
    gA: cute.Tensor,   # (batch, 16, 16) FP32 global
    gB: cute.Tensor,   # (batch, 16, 16) FP32 global
    gC: cute.Tensor,   # (batch, 16, 16) FP32 global
):
    """Test kernel for TF32 16x16 GEMM."""
    tidx, _, _ = cute.arch.thread_idx()
    batch_idx, _, _ = cute.arch.block_idx()
    lane_id = tidx % 32
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    
    # SMEM layout: row-major 16x16
    smem_layout = cute.make_layout((16, 16), stride=(16, 1))
    sA = smem.allocate_tensor(cutlass.Float32, smem_layout, 16)
    sB = smem.allocate_tensor(cutlass.Float32, smem_layout, 16)
    sC = smem.allocate_tensor(cutlass.Float32, smem_layout, 16)
    
    # Cooperative load: each thread loads 8 elements
    # 256 elements / 32 threads = 8 elements per thread
    for rep in range(8):
        idx = tidx + rep * 32
        row = idx // 16
        col = idx % 16
        sA[row, col] = gA[batch_idx, row, col]
        sB[row, col] = gB[batch_idx, row, col]
    
    cute.arch.sync_threads()
    
    # Perform GEMM
    gemm_tf32_16x16(sA, sB, sC, lane_id)
    
    cute.arch.sync_threads()
    
    # Store C from shared to global memory
    for rep in range(8):
        idx = tidx + rep * 32
        row = idx // 16
        col = idx % 16
        gC[batch_idx, row, col] = sC[row, col]


@cute.jit
def test_tf32_gemm_host(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    batch_size: cutlass.Constexpr[int],
):
    """Host function for TF32 GEMM test."""
    grid = (batch_size, 1, 1)
    
    test_tf32_gemm_kernel(A, B, C).launch(
        grid=grid,
        block=(THREADS_PER_CTA, 1, 1),
    )


# ===========================================================================
# Test Function
# ===========================================================================
def test_tf32_gemm():
    import torch
    
    BATCH_SIZE = 1024
    
    print("=" * 60)
    print("Testing TF32 MMA 16x16 GEMM")
    print("=" * 60)
    
    cutlass.cuda.initialize_cuda_context()
    
    torch.manual_seed(42)
    
    # Create test matrices
    A = torch.randn(BATCH_SIZE, 16, 16, device="cuda", dtype=torch.float32)
    B = torch.randn(BATCH_SIZE, 16, 16, device="cuda", dtype=torch.float32)
    C = torch.zeros(BATCH_SIZE, 16, 16, device="cuda", dtype=torch.float32)
    
    # Reference computation
    C_ref = torch.bmm(A, B)
    
    # Convert to CuTe tensors
    A_cute = from_dlpack(A, assumed_align=16)
    B_cute = from_dlpack(B, assumed_align=16)
    C_cute = from_dlpack(C, assumed_align=16)
    
    print("\nCompiling kernel...")
    compiled_kernel = cute.compile(
        test_tf32_gemm_host,
        A_cute,
        B_cute,
        C_cute,
        BATCH_SIZE,
    )
    torch.cuda.synchronize()
    print("Compilation done.")
    
    # Run kernel
    print("\nRunning kernel...")
    compiled_kernel(A_cute, B_cute, C_cute)
    torch.cuda.synchronize()
    
    # Verify results
    print("\nVerifying results...")
    
    # TF32 has reduced precision (19-bit mantissa vs 23-bit for FP32)
    # Expect ~0.1% error due to precision loss
    max_diff = (C - C_ref).abs().max().item()
    mean_diff = (C - C_ref).abs().mean().item()
    
    # Use relative error with larger epsilon to avoid division by small numbers
    rel_error = (C - C_ref).abs() / (C_ref.abs().clamp(min=1.0))
    max_rel_error = rel_error.max().item()
    
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative error (clamp min=1.0): {max_rel_error:.6e}")
    
    # TF32 tolerance: expect small absolute error
    if max_diff < 0.1:  # Allow 0.1 absolute error for TF32
        print("\n[PASSED] Results match within TF32 precision!")
    else:
        print("\n[FAILED] Results differ more than expected")
        print("\nSample comparison (batch 0, first 4x4):")
        print("C_computed:")
        print(C[0, :4, :4])
        print("\nC_reference:")
        print(C_ref[0, :4, :4])
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmarking...")
    print("=" * 60)
    
    WARMUP = 5
    ITERS = 100
    
    for _ in range(WARMUP):
        compiled_kernel(A_cute, B_cute, C_cute)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(ITERS):
        compiled_kernel(A_cute, B_cute, C_cute)
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / ITERS
    
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Time per iteration: {mean_time:.3f} ms")
    print(f"  Throughput: {BATCH_SIZE / mean_time * 1000:.0f} matrices/sec")
    
    return True


if __name__ == "__main__":
    test_tf32_gemm()
