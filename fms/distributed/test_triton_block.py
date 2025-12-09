import torch
import math

# Import your Triton function
from triton_offdiag_block import block_softmax_stats_triton

# Naive reference implementation (matches your Python code)
def block_softmax_stats_naive(Q, K, V, query_indices, key_indices, scale, causal):
    B, H, Q_len, Dk = Q.shape
    K_len = K.shape[2]
    Dv = V.shape[-1]

    if Q_len == 0 or K_len == 0:
        m_block = Q.new_full((B, H, Q_len, 1), float("-inf"))
        l_block = Q.new_zeros((B, H, Q_len, 1))
        z_block = Q.new_zeros((B, H, Q_len, Dv))
        return z_block, l_block, m_block

    scores = torch.matmul(Q / scale, K.transpose(-2, -1))  # [B,H,Q,K]

    if causal:
        # future positions: key_idx > query_idx
        # query_indices: [Q_len], key_indices: [K_len]
        qi = query_indices.view(1, 1, -1, 1)   # [1,1,Q,1]
        kj = key_indices.view(1, 1, 1, -1)     # [1,1,1,K]
        future_mask = (kj > qi)               # [1,1,Q,K]
        scores = scores.masked_fill(future_mask, float('-inf'))

    m_block = scores.max(dim=-1, keepdim=True).values            # [B,H,Q,1]
    exp_scores = torch.exp(scores - m_block)                     # [B,H,Q,K]
    l_block = exp_scores.sum(dim=-1, keepdim=True)               # [B,H,Q,1]
    z_block = torch.matmul(exp_scores, V)                        # [B,H,Q,Dv]
    return z_block, l_block, m_block

def test_kernel(
    B=2, H=3, Q_len=17, K_len=23, D_k=64, D_v=32,
    causal=True, device="cuda"
):
    torch.manual_seed(0)

    Q = torch.randn(B, H, Q_len, D_k, device=device, dtype=torch.float32)
    K = torch.randn(B, H, K_len, D_k, device=device, dtype=torch.float32)
    V = torch.randn(B, H, K_len, D_v, device=device, dtype=torch.float32)


    # global indices (here just 0..Q_len-1, 0..K_len-1)
    query_indices = torch.arange(Q_len, device=device, dtype=torch.long)
    key_indices   = torch.arange(K_len, device=device, dtype=torch.long)

    scale = math.sqrt(D_k)

    # Triton output
    z_tri, l_tri, m_tri = block_softmax_stats_triton(
        Q, K, V,
        query_indices, key_indices,
        scale,
        mask=None,                 # currently unused in kernel
        causal=causal,
        block_q=16,
        block_k=32,
    )

    # Naive reference 
    z_ref, l_ref, m_ref = block_softmax_stats_naive(
        Q.to(torch.float32), 
        K.to(torch.float32),
        V.to(torch.float32),
        query_indices,
        key_indices,
        scale,
        causal=causal
    )

    # Compare
    print("max |z_tri - z_ref| =", (z_tri - z_ref).abs().max().item())
    print("max |l_tri - l_ref| =", (l_tri - l_ref).abs().max().item())
    print("max |m_tri - m_ref| =", (m_tri - m_ref).abs().max().item())

    assert torch.allclose(z_tri, z_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(l_tri, l_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(m_tri, m_ref, atol=1e-2, rtol=1e-2)
    print("Triton kernel matches naive implementation")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Need a CUDA device to test Triton kernel")
    test_kernel()
