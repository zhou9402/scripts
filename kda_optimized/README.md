# KDA Optimized

Optimized KDA (Kernel Delta Attention) forward pass using cuTile + Triton fusion.

## Files

- `forward.py` - Main forward pass entry point
- `cumsum_fused.py` - Fused cumsum and scaling operations (Triton)
- `intra_parallel.py` - Intra-chunk attention computation (from FLA)
- `inter_solve.py` - Inter-chunk solve (from FLA)
- `cutile_kernel.py` - Fused cuTile kernel for W, U, O, S computation
- `test.py` - Correctness and benchmark tests
- `run_profile.py` - NSight profiling script

## Usage

### Run Tests

```bash
cd scripts/kda_optimized
PYTHONPATH=..:$PYTHONPATH python test.py --test all
```

Options:
- `--test {naive,fla,benchmark,all}` - Test to run (default: all)
- `--batch-size` - Batch size for correctness tests (default: 1)
- `--seq-len` - Sequence length for correctness tests (default: 128)
- `--num-heads` - Number of heads for correctness tests (default: 2)
- `--head-dim` - Head dimension (default: 128)
- `--chunk-size` - Chunk size (default: 64)
- `--bench-seq-len` - Sequence length for benchmark (default: 8192)
- `--bench-num-heads` - Number of heads for benchmark (default: 96)

### Run Profiling

```bash
cd scripts/kda_optimized
PYTHONPATH=..:$PYTHONPATH nsys profile -o kda_opt_profile --force-overwrite true python run_profile.py
```

View stats:
```bash
nsys stats kda_opt_profile.nsys-rep --report cuda_gpu_kern_sum
```
