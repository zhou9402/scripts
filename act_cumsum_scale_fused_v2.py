# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Fused Gate Activation + Cumsum + Scaling Kernel for KDA (V2).

Using CuTe DSL with 2-phase algorithm and 2-buffer K,Q pipelining:

- Phase 0: Compute cumsum (forward, 8 rows per iteration)
  - iter 0-3: rows 0-31 → SMEM
  - iter 4-5: rows 32-47 → registers
  - iter 6: rows 48-55 → registers + preload K,Q[56:63] → buf[0]
  - iter 7: rows 56-63 → registers + preload K,Q[48:55] → buf[1]

- Phase 1: Compute outputs (backward, 8 rows per iteration)
  - iter 0: compute rows 56-63 from buf[0] | preload K,Q[40:47] → buf[0]
  - iter 1: compute rows 48-55 from buf[1] | preload K,Q[32:39] → buf[1]
  - ...continues...
  - iter 6: compute rows 8-15 from buf[0]
  - iter 7: compute rows 0-7 from buf[1]

Grid: (cdiv(S, BS), NT, B * H)
Each block processes one [BT, BS] tile.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Configuration
BT = 64          # Chunk size (rows)
BS = 128         # Feature block size per CTA
NUM_WARPS = 4
THREADS_PER_BLOCK = NUM_WARPS * 32  # 128 threads
BLOCK_ROWS = 8   # Rows per iteration
NUM_ITERS = BT // BLOCK_ROWS  # 8 iterations


class ActCumsumScaleFusedV2:
    """Optimized fused kernel using CuTe DSL with 2-buffer K,Q pipelining."""
    
    def __init__(self):
        self.num_threads = THREADS_PER_BLOCK
    
    @cute.jit
    def __call__(
        self,
        mG: cute.Tensor,           # [B, T, H, S] raw gate
        mK: cute.Tensor,           # [B, T, H, S] keys
        mQ: cute.Tensor,           # [B, T, H, S] queries
        mA_log: cute.Tensor,       # [H] log of decay rate
        cumsum_scale: cutlass.Float32,
        attn_scale: cutlass.Float32,
        mG_cumsum: cute.Tensor,    # [B, T, H, S] output
        mK_scaled: cute.Tensor,    # [B, T, H, S] output
        mKg: cute.Tensor,          # [B, T, H, S] output
        mQ_scaled: cute.Tensor,    # [B, T, H, S] output
        mGk_last_exp: cute.Tensor, # [B, NT, H, S] output
        B: cutlass.Constexpr,
        T: cutlass.Constexpr,
        H: cutlass.Constexpr,
        S: cutlass.Constexpr,
    ):
        """Launch the optimized kernel."""
        NT = T // BT
        
        # SMEM layout:
        # - cumsum[32, BS] fp32 = 16 KB
        # - K[32, BS] fp16 = 8 KB
        # - Q[32, BS] fp16 = 8 KB
        # Total: 32 KB
        smem_size = 32 * BS * 4 + 32 * BS * 2 + 32 * BS * 2
        
        # Grid: (cdiv(S, BS), NT, B * H)
        grid_x = (S + BS - 1) // BS
        grid_y = NT
        grid_z = B * H
        
        self.kernel(
            mG, mK, mQ, mA_log,
            cumsum_scale, attn_scale,
            mG_cumsum, mK_scaled, mKg, mQ_scaled, mGk_last_exp,
            B, T, H, S, NT,
        ).launch(
            grid=(grid_x, grid_y, grid_z),
            block=(self.num_threads, 1, 1),
            smem=smem_size,
        )
    
    @cute.kernel
    def kernel(
        self,
        mG: cute.Tensor,           # [B, T, H, S]
        mK: cute.Tensor,           # [B, T, H, S]
        mQ: cute.Tensor,           # [B, T, H, S]
        mA_log: cute.Tensor,       # [H]
        cumsum_scale: cutlass.Float32,
        attn_scale: cutlass.Float32,
        mG_cumsum: cute.Tensor,    # [B, T, H, S]
        mK_scaled: cute.Tensor,    # [B, T, H, S]
        mKg: cute.Tensor,          # [B, T, H, S]
        mQ_scaled: cute.Tensor,    # [B, T, H, S]
        mGk_last_exp: cute.Tensor, # [B, NT, H, S]
        B: cutlass.Constexpr,
        T: cutlass.Constexpr,
        H: cutlass.Constexpr,
        S: cutlass.Constexpr,
        NT: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i_s, i_t, i_bh = cute.arch.block_idx()
        
        i_b = i_bh // H
        i_h = i_bh % H
        
        # Column this thread handles
        col = i_s * BS + tidx
        
        # Chunk start row
        chunk_start = i_t * BT
        
        # =====================================================================
        # Allocate SMEM
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()
        
        # Cumsum first half [32, BS] fp32
        sCumsum = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32, BS)), 16)
        
        # K,Q buffers [32, BS] fp16 each - for second half K,Q
        sK = smem.allocate_tensor(cutlass.Float16, cute.make_layout((32, BS)), 16)
        sQ = smem.allocate_tensor(cutlass.Float16, cute.make_layout((32, BS)), 16)
        
        # =====================================================================
        # Register array for cumsum second half (32 values per thread)
        # =====================================================================
        rCumsum = cute.make_rmem_tensor(cute.make_layout((32,)), cutlass.Float32)
        rCumsum.fill(0.0)
        
        # Accumulator
        acc = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
        acc[0] = cutlass.Float32(0.0)
        
        # Load A_log for this head
        neg_exp_A = -cute.exp(mA_log[i_h])
        
        # =====================================================================
        # Phase 0 Part 1: Cumsum rows 0-31 → SMEM
        # =====================================================================
        for row in cutlass.range_constexpr(32):
            global_row = chunk_start + row
            g_val = mG[i_b, global_row, i_h, col].to(cutlass.Float32)
            softplus_val = cute.log(cutlass.Float32(1.0) + cute.exp(g_val))
            g_activated = neg_exp_A * softplus_val
            acc[0] = acc[0] + g_activated
            sCumsum[row, tidx] = acc[0] * cumsum_scale
        
        # =====================================================================
        # Phase 0 Part 2: Cumsum rows 32-63 → registers
        #                 + Preload K,Q[32:63] → SMEM
        # =====================================================================
        for row in cutlass.range_constexpr(32):
            global_row = chunk_start + 32 + row
            g_val = mG[i_b, global_row, i_h, col].to(cutlass.Float32)
            softplus_val = cute.log(cutlass.Float32(1.0) + cute.exp(g_val))
            g_activated = neg_exp_A * softplus_val
            acc[0] = acc[0] + g_activated
            rCumsum[row] = acc[0] * cumsum_scale
            # Preload K,Q second half (rows 32-63)
            sK[row, tidx] = mK[i_b, global_row, i_h, col]
            sQ[row, tidx] = mQ[i_b, global_row, i_h, col]
        
        # gk_last = cumsum[63]
        gk_last = rCumsum[31]
        
        cute.arch.sync_threads()
        
        # =====================================================================
        # Phase 1 Part 1: Compute rows 32-63 (cumsum in registers, K,Q in SMEM)
        #                 + Preload K,Q[0:31] → SMEM (reuse buffer)
        # =====================================================================
        for row in cutlass.range_constexpr(32):
            cs = rCumsum[row]
            k_val = sK[row, tidx].to(cutlass.Float32)
            q_val = sQ[row, tidx].to(cutlass.Float32)
            
            exp2_cs = cute.exp2(cs)
            exp2_kg = cute.exp2(gk_last - cs)
            
            global_row = chunk_start + 32 + row
            mG_cumsum[i_b, global_row, i_h, col] = cs
            mK_scaled[i_b, global_row, i_h, col] = (k_val * exp2_cs).to(cutlass.Float16)
            mQ_scaled[i_b, global_row, i_h, col] = (q_val * exp2_cs * attn_scale).to(cutlass.Float16)
            mKg[i_b, global_row, i_h, col] = (k_val * exp2_kg).to(cutlass.Float16)
            
            # Preload K,Q first half (rows 0-31) - reuse SMEM buffer
            global_row_first = chunk_start + row
            sK[row, tidx] = mK[i_b, global_row_first, i_h, col]
            sQ[row, tidx] = mQ[i_b, global_row_first, i_h, col]
        
        cute.arch.sync_threads()
        
        # =====================================================================
        # Phase 1 Part 2: Compute rows 0-31 (cumsum in SMEM, K,Q in SMEM)
        # =====================================================================
        for row in cutlass.range_constexpr(32):
            cs = sCumsum[row, tidx]
            k_val = sK[row, tidx].to(cutlass.Float32)
            q_val = sQ[row, tidx].to(cutlass.Float32)
            
            exp2_cs = cute.exp2(cs)
            exp2_kg = cute.exp2(gk_last - cs)
            
            global_row = chunk_start + row
            mG_cumsum[i_b, global_row, i_h, col] = cs
            mK_scaled[i_b, global_row, i_h, col] = (k_val * exp2_cs).to(cutlass.Float16)
            mQ_scaled[i_b, global_row, i_h, col] = (q_val * exp2_cs * attn_scale).to(cutlass.Float16)
            mKg[i_b, global_row, i_h, col] = (k_val * exp2_kg).to(cutlass.Float16)
        
        # Store gk_last_exp
        mGk_last_exp[i_b, i_t, i_h, col] = cute.exp2(gk_last)


# Global cache for compiled kernels
_compiled_kernels = {}


def act_cumsum_scale_fused_v2(
    g: torch.Tensor,           # [B, T, H, S] raw gate
    k: torch.Tensor,           # [B, T, H, S] keys
    q: torch.Tensor,           # [B, T, H, S] queries
    A_log: torch.Tensor,       # [H] log of decay rate
    chunk_size: int,
    cumsum_scale: float = 1.0,
    attn_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized fused gate activation + cumsum + scaling for KDA."""
    global _compiled_kernels
    
    B, T, H, S = g.shape
    
    assert chunk_size == BT, f"This kernel requires chunk_size={BT}, got {chunk_size}"
    assert T % BT == 0, f"T ({T}) must be divisible by chunk_size ({BT})"
    assert S % BS == 0 or S == BS, f"S ({S}) must be divisible by {BS}"
    
    NT = T // BT
    
    # Ensure contiguous
    g = g.contiguous()
    k = k.contiguous()
    q = q.contiguous()
    A_log = A_log.contiguous()
    
    # Allocate outputs
    g_cumsum = torch.empty(B, T, H, S, dtype=torch.float32, device=g.device)
    k_scaled = torch.empty_like(k)
    kg = torch.empty_like(k)
    q_scaled = torch.empty_like(q)
    gk_last_exp = torch.empty(B, NT, H, S, dtype=torch.float32, device=g.device)
    
    # Cache key based on shapes
    cache_key = (B, T, H, S)
    
    if cache_key not in _compiled_kernels:
        kernel_op = ActCumsumScaleFusedV2()
        
        # Compile once
        compiled = cute.compile(
            kernel_op,
            from_dlpack(g),
            from_dlpack(k),
            from_dlpack(q),
            from_dlpack(A_log),
            cutlass.Float32(cumsum_scale),
            cutlass.Float32(attn_scale),
            from_dlpack(g_cumsum),
            from_dlpack(k_scaled),
            from_dlpack(kg),
            from_dlpack(q_scaled),
            from_dlpack(gk_last_exp),
            B, T, H, S,
        )
        _compiled_kernels[cache_key] = compiled
    
    compiled = _compiled_kernels[cache_key]
    
    # Run
    compiled(
        from_dlpack(g),
        from_dlpack(k),
        from_dlpack(q),
        from_dlpack(A_log),
        cutlass.Float32(cumsum_scale),
        cutlass.Float32(attn_scale),
        from_dlpack(g_cumsum),
        from_dlpack(k_scaled),
        from_dlpack(kg),
        from_dlpack(q_scaled),
        from_dlpack(gk_last_exp),
    )
    
    return g_cumsum, k_scaled, kg, q_scaled, gk_last_exp


def test():
    """Test the kernel."""
    torch.manual_seed(42)
    
    # Test with realistic size
    B, T, H, S = 1, 512, 8, BS  # Moderate size for quick test
    chunk_size = BT
    
    g = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda') * 0.1
    k = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda')
    q = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda')
    A_log = torch.zeros(H, dtype=torch.float32, device='cuda')
    
    cumsum_scale = 1.0
    attn_scale = 1.0
    
    print(f"Testing: B={B}, T={T}, H={H}, S={S}")
    print(f"Chunks per sequence: {T // chunk_size}")
    print(f"Total blocks: {(S // BS) * (T // BT) * (B * H)}")
    
    # Run kernel
    print("\nRunning kernel...")
    g_cumsum, k_scaled, kg, q_scaled, gk_last_exp = act_cumsum_scale_fused_v2(
        g, k, q, A_log, chunk_size, cumsum_scale, attn_scale
    )
    
    # Reference - per-chunk cumsum
    print("Computing reference...")
    g_float = g.float()
    g_activated = -1.0 * torch.log(1.0 + torch.exp(g_float))
    
    # Per-chunk cumsum: reshape to [B, NT, chunk_size, H, S], cumsum on dim=2, reshape back
    NT = T // chunk_size
    g_activated_chunked = g_activated.view(B, NT, chunk_size, H, S)
    g_cumsum_chunked = torch.cumsum(g_activated_chunked, dim=2) * cumsum_scale
    g_cumsum_ref = g_cumsum_chunked.view(B, T, H, S)
    
    # gk_last is the last row of each chunk
    gk_last_ref = g_cumsum_chunked[:, :, -1, :, :]  # [B, NT, H, S]
    gk_last_ref_expanded = gk_last_ref.unsqueeze(2).expand(B, NT, chunk_size, H, S).reshape(B, T, H, S)
    
    k_float = k.float()
    q_float = q.float()
    k_scaled_ref = k_float * torch.exp2(g_cumsum_ref)
    q_scaled_ref = q_float * torch.exp2(g_cumsum_ref) * attn_scale
    kg_ref = k_float * torch.exp2(gk_last_ref_expanded - g_cumsum_ref)
    
    # Compare
    print("\nResults:")
    g_diff = (g_cumsum_ref - g_cumsum).abs().max().item()
    k_diff = (k_scaled_ref - k_scaled.float()).abs().max().item()
    kg_diff = (kg_ref - kg.float()).abs().max().item()
    q_diff = (q_scaled_ref - q_scaled.float()).abs().max().item()
    gk_last_diff = (torch.exp2(gk_last_ref) - gk_last_exp).abs().max().item()
    
    print(f"  g_cumsum diff:   {g_diff:.6f}")
    print(f"  k_scaled diff:   {k_diff:.6f}")
    print(f"  kg diff:         {kg_diff:.6f}")
    print(f"  q_scaled diff:   {q_diff:.6f}")
    print(f"  gk_last_exp diff: {gk_last_diff:.6f}")
    
    tol_g = 0.01
    tol_kq = 0.1
    
    if all([g_diff < tol_g, k_diff < tol_kq, kg_diff < tol_kq, q_diff < tol_kq, gk_last_diff < tol_kq]):
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Test failed!")
        # Debug info
        print(f"\nDebug - g_cumsum[0, 0, 0, :8]:")
        print(f"  ref: {g_cumsum_ref[0,0,0,:8].tolist()}")
        print(f"  out: {g_cumsum[0,0,0,:8].tolist()}")
        return False


def benchmark():
    """Benchmark CuTe implementation."""
    import time
    
    torch.manual_seed(42)
    
    # Real workload size
    B, T, H, S = 1, 8192, 96, BS  # S=128 (BS)
    chunk_size = BT
    NT = T // chunk_size
    
    g = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda')
    k = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda')
    q = torch.randn(B, T, H, S, dtype=torch.float16, device='cuda')
    A_log = torch.randn(H, dtype=torch.float32, device='cuda') * 0.1
    
    cumsum_scale = 1.4426950408889634  # RCP_LN2
    attn_scale = 0.125
    
    print(f"Benchmark: B={B}, T={T}, H={H}, S={S}, chunk_size={chunk_size}")
    print(f"Total blocks: {(S // BS) * NT * (B * H)}")
    print()
    
    # =========================================================================
    # Benchmark CuTe v2
    # =========================================================================
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = act_cumsum_scale_fused_v2(g, k, q, A_log, chunk_size, cumsum_scale, attn_scale)
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking...")
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        _ = act_cumsum_scale_fused_v2(g, k, q, A_log, chunk_size, cumsum_scale, attn_scale)
    torch.cuda.synchronize()
    cute_time = (time.time() - start) / iterations * 1000
    print(f"CuTe v2:  {cute_time:.4f} ms per iteration")
    
    # Compute throughput (rough estimate)
    total_bytes = g.numel() * 2 * 8  # inputs + outputs roughly
    throughput = total_bytes / (cute_time / 1000) / 1e9
    print(f"Throughput: ~{throughput:.1f} GB/s")
    
    return cute_time


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        test()

