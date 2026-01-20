"""
Test consistency between Fused GDN (cuda.tile) and FLA (Triton) implementations.

GDN: fused_gdn (single kernel)
FLA: recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o

Compare outputs: O (output) and S (state)

Note: GDN stores S as (B, H, d, d) and uses S^T in computation.
      FLA stores h as (B, NT, H, K, V) where K=V=d.
      So GDN's S = FLA's h^T, meaning S^T = h.
"""

import argparse
import torch
import torch.nn.functional as F
import cuda.tile as ct

from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

ConstInt = ct.Constant[int]
CHUNK_SIZE = 64


def safe_exp(x):
    """Safe exponential: exp(x) if x <= 0, else 0. Prevents overflow."""
    return ct.where(x <= 0.0, ct.exp(x), 0.0)


# ============== Fused GDN Kernel (with W, U, Delta output for debugging) ==============
@ct.kernel
def fused_gdn_kernel(
    S,      # (B, H, d, d) bfloat16 - state (stored as h^T)
    T_in,   # (B, num_chunks, chunk_size, H, chunk_size) bfloat16
    V,      # (B, num_chunks, chunk_size, H, d) bfloat16
    Q,      # (B, num_chunks, chunk_size, H, d) bfloat16
    K,      # (B, num_chunks, chunk_size, H, d) bfloat16
    G,      # (B, num_chunks, chunk_size, H) float32 - gate cumsum
    Beta,   # (B, num_chunks, chunk_size, H) bfloat16
    O,      # (B, num_chunks, chunk_size, H, d) bfloat16 - output
    W_out,  # (B, num_chunks, chunk_size, H, d) bfloat16 - W for debugging
    U_out,  # (B, num_chunks, chunk_size, H, d) bfloat16 - U for debugging
    Delta_out,  # (B, num_chunks, chunk_size, H, d) bfloat16 - delta for debugging
    d: ConstInt, chunk_size: ConstInt, H: ConstInt
):
    """Fused GDN kernel: combines scale_vk and gdn.
    
    S is stored as h^T (transposed relative to FLA's h).
    Uses S^T in computation, which equals FLA's h.
    """
    num_chunks = T_in.shape[1]  # Can be dynamic, only used in for loop
    
    linear_idx = ct.bid(0)
    b_idx = linear_idx // H
    h_idx = linear_idx % H
    zero_pad = ct.PaddingMode.ZERO
    
    # Load S (stored as h^T)
    s = ct.load(S, index=(b_idx, h_idx, 0, 0), shape=(1, 1, d, d), padding_mode=zero_pad)
    s = ct.astype(ct.reshape(s, (d, d)), ct.float32)  # Keep s in float32
    
    # Causal mask
    offs_row = ct.arange(chunk_size, dtype=ct.int32)[:, None]
    offs_col = ct.arange(chunk_size, dtype=ct.int32)[None, :]
    mask = ct.where(offs_row >= offs_col, 1.0, 0.0)
    
    for c in range(num_chunks):
        # Load inputs
        T = ct.reshape(ct.load(T_in, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, chunk_size), padding_mode=zero_pad), (chunk_size, chunk_size))
        k = ct.reshape(ct.load(K, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        v = ct.reshape(ct.load(V, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        q = ct.reshape(ct.load(Q, index=(b_idx, c, 0, h_idx, 0), shape=(1, 1, chunk_size, 1, d), padding_mode=zero_pad), (chunk_size, d))
        g_raw = ct.reshape(ct.load(G, index=(b_idx, c, 0, h_idx), shape=(1, 1, chunk_size, 1), padding_mode=zero_pad), (chunk_size, 1))
        beta = ct.reshape(ct.load(Beta, index=(b_idx, c, 0, h_idx), shape=(1, 1, chunk_size, 1), padding_mode=zero_pad), (1, chunk_size))
        g_chunk_last = ct.reshape(ct.load(G, index=(b_idx, c, chunk_size - 1, h_idx), shape=(1, 1, 1, 1), padding_mode=zero_pad), (1, 1))
        
        # Gate computations
        exp_g = ct.exp(g_raw)
        g_chunk = safe_exp(g_chunk_last - g_raw)
        g_chunk_last_exp = ct.exp(g_chunk_last)
        g_attn = safe_exp(g_raw - ct.transpose(g_raw))
        g_out = ct.exp(g_raw)
        
        # Scale T instead of V/K
        exp_g_t = ct.transpose(exp_g)
        T_v = T * beta
        T_k = ct.astype(T_v * exp_g_t, K.dtype)
        
        # W = T_k @ K, U = T_v @ V
        w = ct.astype(ct.mma(T_k, k, ct.full((chunk_size, d), 0.0, dtype=ct.float32)), K.dtype)
        u = ct.astype(ct.mma(T_v, v, ct.full((chunk_size, d), 0.0, dtype=ct.float32)), K.dtype)
        
        # Save W and U for debugging
        ct.store(W_out, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(w, (1, 1, chunk_size, 1, d)))
        ct.store(U_out, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(u, (1, 1, chunk_size, 1, d)))
        
        # delta = (U - W @ S^T) * g_chunk
        # S^T = h (FLA's state), so this matches FLA
        s_bf16 = ct.astype(s, K.dtype)  # Convert to bfloat16 for mma
        s_t = ct.transpose(s_bf16)
        w_st = ct.mma(w, s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        new_v = u - w_st
        new_v_bf16 = ct.astype(new_v, K.dtype)
        
        # Save new_v (before gate) for debugging - matches FLA's v_new storage
        ct.store(Delta_out, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(new_v_bf16, (1, 1, chunk_size, 1, d)))
        
        delta = new_v * g_chunk
        delta_bf16 = ct.astype(delta, K.dtype)
        
        # O = Q @ S^T * exp(g) + mask(Q @ K^T * g_attn) @ new_v
        # Note: FLA's chunk_fwd_o uses v_new (before gate), NOT delta (after gate)
        o1 = ct.mma(q, s_t, ct.full((chunk_size, d), 0.0, dtype=ct.float32)) * g_out
        qk = ct.mma(q, ct.transpose(k), ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32))
        qk_masked = qk * g_attn * mask
        o2 = ct.mma(ct.astype(qk_masked, K.dtype), new_v_bf16, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        o_out = ct.astype(o1 + o2, O.dtype)
        ct.store(O, index=(b_idx, c, 0, h_idx, 0), tile=ct.reshape(o_out, (1, 1, chunk_size, 1, d)))
        
        # S = S * exp(g_last) + delta^T @ K
        # Note: FLA does h += k^T @ delta, we do s += delta^T @ k
        # Since s = h^T, this is equivalent: s^T += k^T @ delta => s += delta^T @ k
        s_update = ct.mma(ct.transpose(delta_bf16), k, ct.full((d, d), 0.0, dtype=ct.float32))
        s = s * g_chunk_last_exp + s_update
    
    # Store final state (convert to bfloat16)
    s_out = ct.astype(ct.reshape(s, (1, 1, d, d)), S.dtype)
    ct.store(S, index=(b_idx, h_idx, 0, 0), tile=s_out)


def fused_gdn(S, T, G, Beta, V, Q, K, chunk_size=CHUNK_SIZE):
    """Fused GDN forward pass. Returns O, W, U, Delta for debugging."""
    B_dim, seq_len, H, d = K.shape
    num_chunks = seq_len // chunk_size
    
    T_r = T.reshape(B_dim, num_chunks, chunk_size, H, chunk_size)
    V_r = V.reshape(B_dim, num_chunks, chunk_size, H, d)
    Q_r = Q.reshape(B_dim, num_chunks, chunk_size, H, d)
    K_r = K.reshape(B_dim, num_chunks, chunk_size, H, d)
    G_r = G.reshape(B_dim, num_chunks, chunk_size, H)
    Beta_r = Beta.reshape(B_dim, num_chunks, chunk_size, H)
    O_r = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=K.device)
    W_r = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=K.device)
    U_r = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=K.device)
    Delta_r = torch.empty(B_dim, num_chunks, chunk_size, H, d, dtype=K.dtype, device=K.device)
    
    with torch.cuda.device(K.device):
        ct.launch(torch.cuda.current_stream(), (B_dim * H,), fused_gdn_kernel,
                  (S, T_r, V_r, Q_r, K_r, G_r, Beta_r, O_r, W_r, U_r, Delta_r, d, chunk_size, H))
    
    return (O_r.reshape(B_dim, seq_len, H, d), W_r.reshape(B_dim, seq_len, H, d), 
            U_r.reshape(B_dim, seq_len, H, d), Delta_r.reshape(B_dim, seq_len, H, d))


# ============== Test ==============
def compare_tensors(name, t1, t2, atol=1e-2, rtol=1e-2, show_values=True):
    """Compare two tensors."""
    diff = (t1.float() - t2.float()).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    is_close = torch.allclose(t1.float(), t2.float(), atol=atol, rtol=rtol)
    
    total = diff.numel()
    # |diff| < 1e-2
    num_close_2 = (diff < 1e-2).sum().item()
    pct_close_2 = 100.0 * num_close_2 / total
    # |diff| < 1e-3
    num_close_3 = (diff < 1e-3).sum().item()
    pct_close_3 = 100.0 * num_close_3 / total
    
    status = f"{GREEN}✓{RESET}" if is_close else f"{RED}✗{RESET}"
    print(f"  {status} {name}: max={max_diff:.6f}, mean={mean_diff:.6f}")
    print(f"      |diff|<1e-2: {num_close_2}/{total} ({pct_close_2:.2f}%), |diff|<1e-3: {num_close_3}/{total} ({pct_close_3:.2f}%)")
    
    if not is_close and show_values:
        print(f"      {CYAN}FLA[0,0,0,:4]:{RESET}  {t1.flatten()[:4].tolist()}")
        print(f"      {YELLOW}GDN[0,0,0,:4]:{RESET}  {t2.flatten()[:4].tolist()}")
    return is_close


def test_fused_gdn(batch_size=1, seq_len=128, num_heads=2, head_dim=128, chunk_size=64, atol=1e-2, rtol=1e-2):
    device, dtype = "cuda", torch.bfloat16
    
    print(f"{BOLD}{CYAN}Testing Fused GDN vs FLA consistency{RESET}")
    print(f"B={batch_size}, T={seq_len}, H={num_heads}, d={head_dim}, chunk_size={chunk_size}")
    print("-" * 60)
    
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    K = F.normalize(torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device), p=2, dim=-1)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    Beta = torch.rand(batch_size, seq_len, num_heads, dtype=dtype, device=device).sigmoid()
    G_raw = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device))
    
    # FLA initial_state: (B, H, K, V) = (B, H, d, d)
    # GDN S: (B, H, d, d) stored as h^T
    # So if FLA starts with zeros, GDN should also start with zeros (transpose of zeros is zeros)
    S_init = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=dtype, device=device)
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    scale = 1.0
    
    # ========== FLA ==========
    print(f"\n{YELLOW}Running FLA (Triton)...{RESET}")
    G_cumsum = chunk_local_cumsum(G_raw, chunk_size=chunk_size, cu_seqlens=None)
    A = chunk_scaled_dot_kkt_fwd(k=K, beta=Beta, g_cumsum=G_cumsum, cu_seqlens=None, chunk_size=chunk_size, output_dtype=torch.float32)
    A_solved = solve_tril(A=A, cu_seqlens=None, output_dtype=K.dtype)
    w_fla, u_fla = recompute_w_u_fwd(k=K, v=V, beta=Beta, A=A_solved, g_cumsum=G_cumsum, cu_seqlens=None)
    
    S_fla = S_init.clone()
    h_fla, v_new_fla = chunk_gated_delta_rule_fwd_h(k=K, w=w_fla, u=u_fla, g=G_cumsum, initial_state=S_fla,
                                                     initial_state_indices=initial_state_indices, cu_seqlens=None)
    O_fla = chunk_fwd_o(q=Q, k=K, v=v_new_fla, h=h_fla, g=G_cumsum, scale=scale, cu_seqlens=None)
    
    # ========== Fused GDN (cuda.tile) ==========
    print(f"{YELLOW}Running Fused GDN (cuda.tile)...{RESET}")
    # GDN stores S as h^T, so we need to transpose S_init before passing
    # Input: S_gdn = S_init^T (GDN expects transposed state)
    S_gdn = S_init.transpose(-1, -2).clone().contiguous()
    O_gdn, W_gdn, U_gdn, Delta_gdn = fused_gdn(S=S_gdn, T=A_solved, G=G_cumsum, Beta=Beta, V=V, Q=Q, K=K, chunk_size=chunk_size)
    
    # ========== Compare W and U first ==========
    print(f"\n{BOLD}Step 1: Compare W and U:{RESET}")
    w_ok = compare_tensors("W", w_fla, W_gdn, atol, rtol)
    u_ok = compare_tensors("U", u_fla, U_gdn, atol, rtol)
    
    # ========== Compare Delta (v_new) ==========
    print(f"\n{BOLD}Step 2: Compare Delta (v_new):{RESET}")
    # v_new_fla is the delta that FLA passes to chunk_fwd_o
    delta_ok = compare_tensors("Delta (v_new)", v_new_fla, Delta_gdn, atol, rtol)
    
    # ========== Compare O and S ==========
    print(f"\n{BOLD}Step 3: Compare O and S:{RESET}")
    o_ok = compare_tensors("O (output)", O_fla, O_gdn, atol, rtol)
    
    # For state comparison: S_gdn = h^T, so S_gdn^T should equal S_fla (which is h after update)
    S_gdn_as_h = S_gdn.transpose(-1, -2).contiguous()
    s_ok = compare_tensors("S (state, S_gdn^T vs S_fla)", S_fla, S_gdn_as_h, atol, rtol)
    
    all_ok = w_ok and u_ok and delta_ok and o_ok and s_ok
    print()
    print(f"{GREEN}{BOLD}✓ ALL PASSED{RESET}" if all_ok else f"{RED}{BOLD}✗ FAILED{RESET}")
    
    if not all_ok:
        print(f"\n{BOLD}Debug Info:{RESET}")
        print(f"  G_cumsum dtype: {G_cumsum.dtype}")
        print(f"  A_solved dtype: {A_solved.dtype}")
        print(f"  h_fla dtype: {h_fla.dtype}")
        print(f"  O_fla dtype: {O_fla.dtype}, O_gdn dtype: {O_gdn.dtype}")
        print(f"  S_fla dtype: {S_fla.dtype}, S_gdn dtype: {S_gdn.dtype}")
    
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    
    test_fused_gdn(args.batch_size, args.seq_len, args.num_heads, args.head_dim, args.chunk_size, args.atol, args.rtol)
