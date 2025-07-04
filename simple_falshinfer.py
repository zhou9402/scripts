from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np
import torch
from triton.testing import do_bench

import flashinfer


def run_bench(
    kv_lens: Sequence[int],
    qo_lens: Sequence[int],
    *,
    page_block_size: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    device: int = 0,
    causal: bool = True,
) -> Tuple[float, float, float, float, float]:
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_data = torch.randn(
        num_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # old
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="flash",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks, dtype=torch.int32, device=device),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    ms_old = do_bench(lambda: wrapper_old.run(q, kv_data))

    return ms_old


def synthesize_seq_len_configs() -> List[List[Tuple[int, int]]]:
    cfgs: List[List[Tuple[int, int]]] = [
        [(4096, 1)] * 128,  # decode-only
    ]
    return cfgs


def main() -> None:
    np.random.seed(42)
    torch.random.manual_seed(42)

    seq_len_cfgs = synthesize_seq_len_configs()

    sweep = {
        "page_block_size": (1, 8, 16),
        "head_dim": (128,),
        "num_kv_heads": (8,),
        "num_qo_heads": (64,),
    }

    for cfg_id, pairs in enumerate(seq_len_cfgs, start=1):
        kv_lens = [p[0] for p in pairs]
        qo_lens = [p[1] for p in pairs]
        for pbs, hd, n_kv, n_qo in itertools.product(
            sweep["page_block_size"],
            sweep["head_dim"],
            sweep["num_kv_heads"],
            sweep["num_qo_heads"],
        ):

            ms_old = run_bench(
                kv_lens,
                qo_lens,
                page_block_size=pbs,
                num_kv_heads=n_kv,
                num_qo_heads=n_qo,
                head_dim=hd,
                device=0,
                causal=True,
            )
            print(f"scheduler: BatchPrefillWithPagedKVCacheWrapper")
            print(f"seq_cfg_id: {cfg_id}")
            print(f"page_size: {pbs}")
            print(f"head_dim: {hd}")
            print(f"num_kv_heads: {n_kv}")
            print(f"num_qo_heads: {n_qo}")
            print(f"time_ms: {ms_old}")
            print("---")


if __name__ == "__main__":
    main()
