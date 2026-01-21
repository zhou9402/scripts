#!/usr/bin/env python3
"""Test precision of solve_tril_cute_v2.py with multiple test cases"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from solve_tril_cute_v2 import solve_tril_host_v2, BLOCK_SIZE


def test_single_batch(A, name, threshold=0.01):
    """Test a single batch of matrices"""
    BATCH_SIZE = A.shape[0]
    
    # PyTorch reference
    A_inv_torch = torch.zeros_like(A)
    for i in range(BATCH_SIZE):
        A_full = A[i] + torch.eye(64, device="cuda", dtype=torch.float16)
        A_inv_torch[i] = torch.linalg.inv(A_full.float()).half()
    
    # CuTe V2
    Ai_cute = torch.zeros_like(A)
    A_cute = from_dlpack(A, assumed_align=16)
    Ai_cute_tensor = from_dlpack(Ai_cute, assumed_align=16)
    
    compiled = cute.compile(solve_tril_host_v2, A_cute, Ai_cute_tensor, BATCH_SIZE)
    compiled(A_cute, Ai_cute_tensor)
    torch.cuda.synchronize()
    
    # Block positions
    blocks = [
        ("Ai11", 0, 0), ("Ai22", 16, 16), ("Ai33", 32, 32), ("Ai44", 48, 48),
        ("Ai21", 16, 0), ("Ai32", 32, 16), ("Ai43", 48, 32),
        ("Ai31", 32, 0), ("Ai42", 48, 16), ("Ai41", 48, 0),
    ]
    
    # Collect errors across all batches
    all_max_errors = []
    all_mean_errors = []
    block_errors = {name: [] for name, _, _ in blocks}
    
    for b in range(BATCH_SIZE):
        for block_name, row, col in blocks:
            cute_block = Ai_cute[b, row:row+16, col:col+16]
            torch_block = A_inv_torch[b, row:row+16, col:col+16]
            
            diff = (cute_block - torch_block).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            block_errors[block_name].append(max_diff)
            all_max_errors.append(max_diff)
            all_mean_errors.append(mean_diff)
    
    overall_max = max(all_max_errors)
    overall_mean = sum(all_mean_errors) / len(all_mean_errors)
    passed = overall_max < threshold
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {name}: max={overall_max:.6f}, mean={overall_mean:.6f} - {status}")
    
    return passed, overall_max, block_errors


def test_precision():
    print("=" * 70)
    print("Comprehensive Precision Test for solve_tril_cute_v2")
    print("=" * 70)
    
    cutlass.cuda.initialize_cuda_context()
    
    all_passed = True
    all_max_errors = []
    
    # =========================================================================
    # Test Group 1: Different random seeds
    # =========================================================================
    print("\n[Group 1] Different Random Seeds (scale=0.1)")
    print("-" * 50)
    
    for seed in [42, 123, 456, 789, 1024, 2048]:
        torch.manual_seed(seed)
        A = torch.randn(16, 64, 64, device="cuda", dtype=torch.float16) * 0.1
        A = A.tril(-1)
        passed, max_err, _ = test_single_batch(A, f"seed={seed}")
        all_passed &= passed
        all_max_errors.append(max_err)
    
    # =========================================================================
    # Test Group 2: Different value scales
    # =========================================================================
    print("\n[Group 2] Different Value Scales")
    print("-" * 50)
    
    torch.manual_seed(42)
    scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    for scale in scales:
        A = torch.randn(16, 64, 64, device="cuda", dtype=torch.float16) * scale
        A = A.tril(-1)
        passed, max_err, _ = test_single_batch(A, f"scale={scale}")
        all_passed &= passed
        all_max_errors.append(max_err)
    
    # =========================================================================
    # Test Group 3: Larger batch sizes
    # =========================================================================
    print("\n[Group 3] Larger Batch Sizes (scale=0.1)")
    print("-" * 50)
    
    torch.manual_seed(42)
    for batch_size in [32, 64, 128, 256]:
        A = torch.randn(batch_size, 64, 64, device="cuda", dtype=torch.float16) * 0.1
        A = A.tril(-1)
        passed, max_err, _ = test_single_batch(A, f"batch={batch_size}")
        all_passed &= passed
        all_max_errors.append(max_err)
    
    # =========================================================================
    # Test Group 4: Special matrices
    # =========================================================================
    print("\n[Group 4] Special Matrices")
    print("-" * 50)
    
    # 4.1: Very sparse (mostly zeros)
    A = torch.zeros(16, 64, 64, device="cuda", dtype=torch.float16)
    for i in range(16):
        for j in range(1, 64):
            A[i, j, j-1] = torch.randn(1).item() * 0.1  # Only sub-diagonal
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "sub-diagonal only")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 4.2: Band matrix (tri-diagonal style)
    torch.manual_seed(42)
    A = torch.zeros(16, 64, 64, device="cuda", dtype=torch.float16)
    for i in range(16):
        for j in range(64):
            for k in range(max(0, j-3), j):  # 3-band lower triangular
                A[i, j, k] = torch.randn(1).item() * 0.1
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "3-band lower tri")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 4.3: Block diagonal pattern
    torch.manual_seed(42)
    A = torch.zeros(16, 64, 64, device="cuda", dtype=torch.float16)
    for i in range(16):
        for block in range(4):
            start = block * 16
            end = start + 16
            A[i, start:end, start:end] = torch.randn(16, 16, device="cuda", dtype=torch.float16) * 0.1
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "block-diagonal")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 4.4: Uniform values
    A = torch.ones(16, 64, 64, device="cuda", dtype=torch.float16) * 0.05
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "uniform 0.05")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 4.5: Small uniform values
    A = torch.ones(16, 64, 64, device="cuda", dtype=torch.float16) * 0.01
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "uniform 0.01")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # =========================================================================
    # Test Group 5: Edge cases
    # =========================================================================
    print("\n[Group 5] Edge Cases")
    print("-" * 50)
    
    # 5.1: All zeros (identity inverse)
    A = torch.zeros(16, 64, 64, device="cuda", dtype=torch.float16)
    passed, max_err, _ = test_single_batch(A, "all zeros (I inverse)")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 5.2: Very small values
    torch.manual_seed(42)
    A = torch.randn(16, 64, 64, device="cuda", dtype=torch.float16) * 0.001
    A = A.tril(-1)
    passed, max_err, _ = test_single_batch(A, "tiny values (0.001)", threshold=0.001)
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # 5.3: Mixed positive/negative with specific pattern
    torch.manual_seed(42)
    A = torch.randn(16, 64, 64, device="cuda", dtype=torch.float16) * 0.1
    A = A.tril(-1)
    # Make some blocks have different scales
    A[:, 16:32, 0:16] *= 2  # A21 block larger
    A[:, 48:64, 32:48] *= 0.5  # A43 block smaller
    passed, max_err, _ = test_single_batch(A, "mixed scales per block")
    all_passed &= passed
    all_max_errors.append(max_err)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total test groups: 5")
    print(f"  Overall max error: {max(all_max_errors):.6f}")
    print(f"  Overall min error: {min(all_max_errors):.6f}")
    print(f"  Mean of max errors: {sum(all_max_errors)/len(all_max_errors):.6f}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return all_passed


def test_detailed_block_analysis():
    """Detailed per-block analysis for debugging"""
    print("\n" + "=" * 70)
    print("Detailed Per-Block Analysis")
    print("=" * 70)
    
    cutlass.cuda.initialize_cuda_context()
    
    torch.manual_seed(42)
    BATCH_SIZE = 4
    A = torch.randn(BATCH_SIZE, 64, 64, device="cuda", dtype=torch.float16) * 0.1
    A = A.tril(-1)
    
    # PyTorch reference
    A_inv_torch = torch.zeros_like(A)
    for i in range(BATCH_SIZE):
        A_full = A[i] + torch.eye(64, device="cuda", dtype=torch.float16)
        A_inv_torch[i] = torch.linalg.inv(A_full.float()).half()
    
    # CuTe V2
    Ai_cute = torch.zeros_like(A)
    A_cute = from_dlpack(A, assumed_align=16)
    Ai_cute_tensor = from_dlpack(Ai_cute, assumed_align=16)
    
    compiled = cute.compile(solve_tril_host_v2, A_cute, Ai_cute_tensor, BATCH_SIZE)
    compiled(A_cute, Ai_cute_tensor)
    torch.cuda.synchronize()
    
    blocks = [
        ("Ai11", 0, 0), ("Ai22", 16, 16), ("Ai33", 32, 32), ("Ai44", 48, 48),
        ("Ai21", 16, 0), ("Ai32", 32, 16), ("Ai43", 48, 32),
        ("Ai31", 32, 0), ("Ai42", 48, 16), ("Ai41", 48, 0),
    ]
    
    print(f"\nAnalyzing {BATCH_SIZE} samples:")
    print("-" * 70)
    print(f"{'Block':<8} {'Max Error':<12} {'Mean Error':<12} {'Std Error':<12} {'Non-zero%':<10}")
    print("-" * 70)
    
    for block_name, row, col in blocks:
        errors = []
        nonzero_pcts = []
        for b in range(BATCH_SIZE):
            cute_block = Ai_cute[b, row:row+16, col:col+16]
            torch_block = A_inv_torch[b, row:row+16, col:col+16]
            
            diff = (cute_block - torch_block).abs()
            errors.extend(diff.flatten().tolist())
            
            nonzero = (torch_block.abs() > 1e-6).sum().item()
            nonzero_pcts.append(nonzero / 256 * 100)
        
        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        std_err = (sum((e - mean_err)**2 for e in errors) / len(errors)) ** 0.5
        avg_nonzero = sum(nonzero_pcts) / len(nonzero_pcts)
        
        print(f"{block_name:<8} {max_err:<12.6f} {mean_err:<12.6f} {std_err:<12.6f} {avg_nonzero:<10.1f}")
    
    print("-" * 70)


if __name__ == "__main__":
    test_precision()
    test_detailed_block_analysis()

