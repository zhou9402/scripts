# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Batched Operations Benchmark: cuTile vs PyTorch vs Triton

All operations use 256x256 matrices, loaded and computed at once (no tiling).
Batch size = 32
"""

import torch
import cuda.tile as ct
import triton
import triton.language as tl

ConstInt = ct.Constant[int]
BATCH_SIZE = 256
DIM = 128


# ============================================================
# Triton Kernels (256x256, one block per batch)
# ============================================================

@triton.jit
def triton_batch_vec_mat_mul_kernel(
    v_ptr, M_ptr, y_ptr,
    stride_vb, stride_vk,
    stride_mb, stride_mk, stride_mn,
    stride_yb, stride_yn,
    DIM: tl.constexpr,
):
    """y[b] = v[b] @ M[b], one block per batch"""
    b_idx = tl.program_id(0)
    
    # Load v[b]: (256,)
    v_offs = b_idx * stride_vb + tl.arange(0, DIM) * stride_vk
    v = tl.load(v_ptr + v_offs)
    
    # Load M[b]: (256, 256)
    k_offs = tl.arange(0, DIM)
    n_offs = tl.arange(0, DIM)
    m_offs = b_idx * stride_mb + k_offs[:, None] * stride_mk + n_offs[None, :] * stride_mn
    m = tl.load(M_ptr + m_offs)
    
    # y = v @ M: (256,) @ (256, 256) = (256,)
    result = tl.sum(v[:, None] * m, axis=0)
    
    # Store y[b]: (256,)
    y_offs = b_idx * stride_yb + n_offs * stride_yn
    tl.store(y_ptr + y_offs, result)


def triton_batch_vec_mat_mul(v, M):
    B, K = v.shape
    _, _, N = M.shape
    y = torch.empty(B, N, device=v.device, dtype=v.dtype)
    grid = (B,)
    triton_batch_vec_mat_mul_kernel[grid](
        v, M, y,
        v.stride(0), v.stride(1),
        M.stride(0), M.stride(1), M.stride(2),
        y.stride(0), y.stride(1),
        DIM=K,
    )
    return y


@triton.jit
def triton_batch_gemv_kernel(
    Mat_ptr, v_ptr, y_ptr,
    stride_mb, stride_mm, stride_mk,
    stride_vb, stride_vk,
    stride_yb, stride_ym,
    DIM: tl.constexpr,
):
    """y[b] = Mat[b] @ v[b], one block per batch"""
    b_idx = tl.program_id(0)
    
    # Load v[b]: (256,)
    v_offs = b_idx * stride_vb + tl.arange(0, DIM) * stride_vk
    v = tl.load(v_ptr + v_offs)
    
    # Load Mat[b]: (256, 256)
    m_offs = tl.arange(0, DIM)
    k_offs = tl.arange(0, DIM)
    mat_offs = b_idx * stride_mb + m_offs[:, None] * stride_mm + k_offs[None, :] * stride_mk
    mat = tl.load(Mat_ptr + mat_offs)
    
    # y = Mat @ v: (256, 256) @ (256,) = (256,)
    result = tl.sum(mat * v[None, :], axis=1)
    
    # Store y[b]: (256,)
    y_offs = b_idx * stride_yb + m_offs * stride_ym
    tl.store(y_ptr + y_offs, result)


def triton_batch_gemv(Mat, v):
    B, M, K = Mat.shape
    y = torch.empty(B, M, device=Mat.device, dtype=Mat.dtype)
    grid = (B,)
    triton_batch_gemv_kernel[grid](
        Mat, v, y,
        Mat.stride(0), Mat.stride(1), Mat.stride(2),
        v.stride(0), v.stride(1),
        y.stride(0), y.stride(1),
        DIM=M,
    )
    return y


@triton.jit
def triton_batch_transpose_kernel(
    A_ptr, B_ptr,
    stride_ab, stride_am, stride_an,
    stride_bb, stride_bn, stride_bm,
    DIM: tl.constexpr,
):
    """B[b] = A[b].T, one block per batch"""
    b_idx = tl.program_id(0)
    
    m_offs = tl.arange(0, DIM)
    n_offs = tl.arange(0, DIM)
    
    # Load A[b]: (256, 256)
    a_offs = b_idx * stride_ab + m_offs[:, None] * stride_am + n_offs[None, :] * stride_an
    a = tl.load(A_ptr + a_offs)
    
    # Transpose
    a_t = tl.trans(a)
    
    # Store B[b]: (256, 256)
    b_offs = b_idx * stride_bb + n_offs[:, None] * stride_bn + m_offs[None, :] * stride_bm
    tl.store(B_ptr + b_offs, a_t)


def triton_batch_transpose(A):
    B, M, N = A.shape
    B_out = torch.empty(B, N, M, device=A.device, dtype=A.dtype)
    grid = (B,)
    triton_batch_transpose_kernel[grid](
        A, B_out,
        A.stride(0), A.stride(1), A.stride(2),
        B_out.stride(0), B_out.stride(1), B_out.stride(2),
        DIM=M,
    )
    return B_out


@triton.jit
def triton_batch_transpose_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_ab, stride_ak, stride_am,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    DIM: tl.constexpr,
):
    """C[b] = A[b].T @ B[b], A:(B,128,128), B:(B,128,128), C:(B,128,128), one block per batch"""
    b_idx = tl.program_id(0)
    
    m_offs = tl.arange(0, DIM)
    k_offs = tl.arange(0, DIM)
    n_offs = tl.arange(0, DIM)
    
    # Load A[b]: (128, 128) then transpose to (128, 128)
    a_offs = b_idx * stride_ab + k_offs[:, None] * stride_ak + m_offs[None, :] * stride_am
    a = tl.load(A_ptr + a_offs)
    a_t = tl.trans(a)  # (128, 128)
    
    # Load B[b]: (128, 128)
    b_offs = b_idx * stride_bb + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
    b = tl.load(B_ptr + b_offs)
    
    # C = A.T @ B: (128, 128) @ (128, 128) = (128, 128)
    c = tl.dot(a_t, b)
    
    # Store C[b]: (128, 128)
    c_offs = b_idx * stride_cb + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.store(C_ptr + c_offs, c.to(tl.float16))


def triton_batch_transpose_gemm(A, B_in):
    B, K, M = A.shape
    _, _, N = B_in.shape
    C = torch.empty(B, M, N, device=A.device, dtype=A.dtype)
    grid = (B,)
    triton_batch_transpose_gemm_kernel[grid](
        A, B_in, C,
        A.stride(0), A.stride(1), A.stride(2),
        B_in.stride(0), B_in.stride(1), B_in.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        DIM=K,
    )
    return C


# ============================================================
# cuTile Kernels (256x256, one block per batch)
# ============================================================

# Case 0: Batch Vector-Matrix Multiplication
# y[b] = v[b] @ M[b], v: (B, 256), M: (B, 256, 256), y: (B, 256)
@ct.kernel
def batch_vec_mat_mul_kernel(v, M, y, Batch: ConstInt):
    """Each block handles one batch item, full 256x256 matrix"""
    bid = ct.bid(0)
    
    # Load v[b]: (256,)
    vec = ct.load(v, index=(bid, 0), shape=(1, DIM))
    vec = ct.reshape(vec, (DIM,))
    
    # Load M[b]: (256, 256)
    m_tile = ct.load(M, index=(bid, 0, 0), shape=(1, DIM, DIM))
    m_tile = ct.reshape(m_tile, (DIM, DIM))
    
    # Compute in fp32
    vec_f32 = ct.astype(vec, ct.float32)
    m_tile_f32 = ct.astype(m_tile, ct.float32)
    
    # y = v @ M: broadcast (256, 1) * (256, 256) -> sum axis=0
    vec_2d = ct.reshape(vec_f32, (DIM, 1))
    prod = vec_2d * m_tile_f32
    result_f32 = ct.sum(prod, axis=0)
    
    result = ct.astype(result_f32, y.dtype)
    ct.store(y, index=(bid, 0), tile=ct.reshape(result, (1, DIM)))


def batch_vec_mat_mul(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """v: (B, 256), M: (B, 256, 256) -> y: (B, 256)"""
    B, _ = v.shape
    y = torch.empty(B, DIM, device=v.device, dtype=v.dtype)
    grid = (B,)
    ct.launch(torch.cuda.current_stream(), grid, batch_vec_mat_mul_kernel,
              (v, M, y, B))
    return y


# Case 1: Batch GEMV (Matrix-Vector Multiplication)
# y[b] = M[b] @ v[b], M: (B, 256, 256), v: (B, 256), y: (B, 256)
@ct.kernel
def batch_gemv_kernel(Mat, v, y, Batch: ConstInt):
    """Each block handles one batch item, full 256x256 matrix"""
    bid = ct.bid(0)

    # Load v[b]: (256,)
    vec = ct.load(v, index=(bid, 0), shape=(1, DIM))
    vec = ct.reshape(vec, (DIM,))
    
    # Load M[b]: (256, 256)
    m_tile = ct.load(Mat, index=(bid, 0, 0), shape=(1, DIM, DIM))
    m_tile = ct.reshape(m_tile, (DIM, DIM))
    
    # Compute in fp32
    m_tile_f32 = ct.astype(m_tile, ct.float32)
    vec_f32 = ct.astype(vec, ct.float32)
    
    # y = M @ v: broadcast (256, 256) * (1, 256) -> sum axis=1
    vec_2d = ct.reshape(vec_f32, (1, DIM))
    prod = m_tile_f32 * vec_2d
    result_f32 = ct.sum(prod, axis=1)
    
    result = ct.astype(result_f32, y.dtype)
    ct.store(y, index=(bid, 0), tile=ct.reshape(result, (1, DIM)))


def batch_gemv(Mat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """y = M @ v, M: (B, 256, 256), v: (B, 256) -> y: (B, 256)"""
    B, _, _ = Mat.shape
    y = torch.empty(B, DIM, device=Mat.device, dtype=Mat.dtype)
    grid = (B,)
    ct.launch(torch.cuda.current_stream(), grid, batch_gemv_kernel,
              (Mat, v, y, B))
    return y


# Case 2: Batch Transpose
# A: (B, 256, 256) -> B_out: (B, 256, 256)
@ct.kernel
def batch_transpose_kernel(A, B_out, Batch: ConstInt):
    """Each block handles one batch item, full 256x256 matrix"""
    bid = ct.bid(0)
    
    # Load A[b]: (256, 256)
    a_tile = ct.load(A, index=(bid, 0, 0), shape=(1, DIM, DIM))
    a_tile = ct.reshape(a_tile, (DIM, DIM))
    
    # Transpose: (256, 256) -> (256, 256)
    b_tile = ct.transpose(a_tile)
    
    # Store B_out[b]: (256, 256)
    ct.store(B_out, index=(bid, 0, 0), tile=ct.reshape(b_tile, (1, DIM, DIM)))


def batch_transpose(A: torch.Tensor) -> torch.Tensor:
    """A: (B, 256, 256) -> B_out: (B, 256, 256)"""
    B, M, N = A.shape
    B_out = torch.empty(B, N, M, device=A.device, dtype=A.dtype)
    grid = (B,)
    ct.launch(torch.cuda.current_stream(), grid, batch_transpose_kernel,
              (A, B_out, B))
    return B_out


# Case 3: Batch Transpose + GEMM
# C[b] = A[b].T @ B[b]
# A: (B, 128, 128), B_in: (B, 128, 128), C: (B, 128, 128)
# Use 128x128 to fit in shared memory
GEMM_DIM = 128

@ct.kernel
def batch_transpose_gemm_kernel(A, B_in, C, Batch: ConstInt):
    """Each block handles one batch item, full 128x128 matrices"""
    bid = ct.bid(0)
    
    # Load A[b]: (128, 128), then transpose to (128, 128)
    a_tile = ct.load(A, index=(bid, 0, 0), shape=(1, DIM, DIM))
    a_tile = ct.reshape(a_tile, (DIM, DIM))
    a_tile_t = ct.transpose(a_tile)  # (128, 128)
    
    # Load B_in[b]: (128, 128)
    b_tile = ct.load(B_in, index=(bid, 0, 0), shape=(1, DIM, DIM))
    b_tile = ct.reshape(b_tile, (DIM, DIM))
    
    # MMA: C = A.T @ B, (128, 128) @ (128, 128) = (128, 128)
    acc = ct.full((DIM, DIM), 0.0, dtype=ct.float32)
    acc = ct.mma(a_tile_t, b_tile, acc)
    
    # Store C[b]: (128, 128)
    c_tile = ct.astype(acc, C.dtype)
    ct.store(C, index=(bid, 0, 0), tile=ct.reshape(c_tile, (1, DIM, DIM)))


def batch_transpose_gemm(A: torch.Tensor, B_in: torch.Tensor) -> torch.Tensor:
    """C = A.T @ B, A: (B, 128, 128), B_in: (B, 128, 128), C: (B, 128, 128)"""
    B, K, M = A.shape
    C = torch.empty(B, M, M, device=A.device, dtype=A.dtype)
    grid = (B,)
    ct.launch(torch.cuda.current_stream(), grid, batch_transpose_gemm_kernel,
              (A, B_in, C, B))
    return C


# ============================================================
# Benchmark Utilities
# ============================================================
def benchmark(name, cutile_fn, pytorch_fn, triton_fn, iters=1000, warmup=10):
    """Benchmark PyTorch vs Triton vs cuTile"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")
    
    # Warmup
    for _ in range(warmup):
        pytorch_fn()
        triton_fn()
        cutile_fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # PyTorch benchmark
    start.record()
    for _ in range(iters):
        pytorch_fn()
    end.record()
    torch.cuda.synchronize()
    pytorch_time_us = start.elapsed_time(end) / iters * 1000
    
    # Triton benchmark
    start.record()
    for _ in range(iters):
        triton_fn()
    end.record()
    torch.cuda.synchronize()
    triton_time_us = start.elapsed_time(end) / iters * 1000
    
    # cuTile benchmark
    start.record()
    for _ in range(iters):
        cutile_fn()
    end.record()
    torch.cuda.synchronize()
    cutile_time_us = start.elapsed_time(end) / iters * 1000
    
    # Find fastest
    times = {'PyTorch': pytorch_time_us, 'Triton': triton_time_us, 'cuTile': cutile_time_us}
    fastest = min(times, key=times.get)
    fastest_time = times[fastest]
    
    print(f"PyTorch: {pytorch_time_us:8.2f} us  ({fastest_time/pytorch_time_us:.2f}x)")
    print(f"Triton:  {triton_time_us:8.2f} us  ({fastest_time/triton_time_us:.2f}x)")
    print(f"cuTile:  {cutile_time_us:8.2f} us  ({fastest_time/cutile_time_us:.2f}x)")
    print(f"Fastest: {fastest}")
    
    return pytorch_time_us, triton_time_us, cutile_time_us


if __name__ == "__main__":
    device = 'cuda'
    dtype = torch.float16
    B = BATCH_SIZE  # 256
    
    print("="*60)
    print(f"cuTile vs PyTorch vs Triton Benchmark [Batch={B}, DIM={DIM}]")
    print("="*60)
    
    # ============================================================
    # Case 0: Batch Vector-Matrix Multiplication (v @ M)
    # v: (B, 256), M: (B, 256, 256), y: (B, 256)
    # ============================================================
    v = torch.randn(B, DIM, dtype=dtype, device=device)
    M = torch.randn(B, DIM, DIM, dtype=dtype, device=device)
    
    benchmark(
        f"Batch VecMatMul v@M (B={B}, {DIM}x{DIM})",
        lambda: batch_vec_mat_mul(v, M),
        lambda: torch.einsum('bk,bkn->bn', v, M),
        lambda: triton_batch_vec_mat_mul(v, M),
    )
    
    # ============================================================
    # Case 1: Batch GEMV (Matrix-Vector Multiplication, M @ v)
    # M: (B, 256, 256), v: (B, 256), y: (B, 256)
    # ============================================================
    Mat_gemv = torch.randn(B, DIM, DIM, dtype=dtype, device=device)
    v_gemv = torch.randn(B, DIM, dtype=dtype, device=device)
    
    benchmark(
        f"Batch GEMV M@v (B={B}, 256x256)",
        lambda: batch_gemv(Mat_gemv, v_gemv),
        lambda: torch.einsum('bij,bj->bi', Mat_gemv, v_gemv),
        lambda: triton_batch_gemv(Mat_gemv, v_gemv),
    )
    
    # ============================================================
    # Case 2: Batch Transpose
    # A: (B, 256, 256) -> B_out: (B, 256, 256)
    # ============================================================
    A_trans = torch.randn(B, DIM, DIM, dtype=dtype, device=device)
    
    benchmark(
        f"Batch Transpose (B={B}, 256x256)",
        lambda: batch_transpose(A_trans),
        lambda: A_trans.transpose(-1, -2).contiguous(),
        lambda: triton_batch_transpose(A_trans),
    )
    
    # ============================================================
    # Case 3: Batch Transpose + GEMM (C = A.T @ B)
    # A: (B, 128, 128), B_in: (B, 128, 128), C: (B, 128, 128)
    # Use 128x128 to fit in shared memory for Triton
    # ============================================================
    A_gemm = torch.randn(B, GEMM_DIM, GEMM_DIM, dtype=dtype, device=device)
    B_gemm = torch.randn(B, GEMM_DIM, GEMM_DIM, dtype=dtype, device=device)
    
    benchmark(
        f"Batch Transpose+GEMM (B={B}, 128x128)",
        lambda: batch_transpose_gemm(A_gemm, B_gemm),
        lambda: torch.bmm(A_gemm.transpose(-1, -2), B_gemm),
        lambda: triton_batch_transpose_gemm(A_gemm, B_gemm),
    )
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)

