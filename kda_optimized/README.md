# KDA Optimized

Optimized KDA (Key-wise Decay Attention) forward pass using cuTile + Triton fusion.

## Features

- **Fused gate activation**: Option to fuse gate activation (sigmoid/softplus) into cumsum kernel
- **Safe gate mode**: Bounded gate values for numerical stability and TensorCore acceleration
- **cuTile fusion**: Fused W, U, O, S computation using cuTile
- **~1.15x-1.24x speedup** over FLA's chunk_kda

## Files

- `forward.py` - Main forward pass entry point
- `cumsum_fused.py` - Fused cumsum and scaling operations (Triton)
- `act_cumsum_scale_fused.py` - Fused gate activation + cumsum + scaling (Triton)
- `intra_parallel.py` - Intra-chunk attention computation (from FLA)
- `inter_solve.py` - Inter-chunk solve (from FLA)
- `cutile_kernel.py` - Fused cuTile kernel for W, U, O, S computation
- `test.py` - Correctness and benchmark tests
- `run_profile.py` - NSight profiling script

## Usage

### Run Tests

```bash
cd scripts/kda_optimized
PYTHONPATH=/path/to/flash-linear-attention:/path/to/scripts:$PYTHONPATH python test.py
```

**Default mode**: `use_gate_in_kernel=True, safe_gate=True` (fused sigmoid activation)

Options:
- `--test {naive,fla,benchmark,all}` - Test to run (default: all)
- `--batch-size` - Batch size (default: 1)
- `--seq-len` - Sequence length (default: 8192)
- `--num-heads` - Number of heads (default: 96)
- `--head-dim` - Head dimension (default: 128)
- `--chunk-size` - Chunk size (default: 64)
- `--no-gate-in-kernel` - Disable fused gate activation (use pre-activated g)
- `--no-safe-gate` - Use softplus activation instead of sigmoid

### Examples

```bash
# Default: fused activation + safe gate (sigmoid)
python test.py --test all

# Fused activation with softplus (no safe gate)
python test.py --test all --no-safe-gate

# Disable fused activation (pre-activated g, original behavior)
python test.py --test all --no-gate-in-kernel
```

### API

```python
from kda_optimized import kda_forward

# With fused activation (default)
O, S = kda_forward(
    q, k, v, g, beta,
    scale=scale,
    initial_state=h0,
    output_final_state=True,
    use_gate_in_kernel=True,   # Fuse gate activation
    A_log=A_log,               # [H] decay rate (required)
    dt_bias=dt_bias,           # [H*K] optional bias
    safe_gate=True,            # Use sigmoid activation
    lower_bound=-5.0,          # Bound for safe_gate
)

# Without fused activation (g is pre-activated)
O, S = kda_forward(
    q, k, v, g_activated, beta,
    scale=scale,
    initial_state=h0,
    output_final_state=True,
    use_gate_in_kernel=False,
)
```

### Run Profiling

```bash
cd scripts/kda_optimized
PYTHONPATH=..:$PYTHONPATH nsys profile -o kda_opt_profile --force-overwrite true python run_profile.py
```

View stats:
```bash
nsys stats kda_opt_profile.nsys-rep --report cuda_gpu_kern_sum
```
