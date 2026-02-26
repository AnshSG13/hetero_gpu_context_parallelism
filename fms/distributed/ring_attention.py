"""
Ring Attention with Heterogeneous GPU Support.

This module implements distributed ring attention for long-context LLM inference,
with support for heterogeneous GPU configurations. Key features:

- Uneven token partitioning across ranks based on GPU capabilities
- Online softmax for correct attention merging across variable-sized shards
- Async P2P communication overlapped with compute via separate CUDA streams
- Custom Triton kernels for block-wise attention statistics

The main entry point is `ring_attention()`, which is called from LLaMABlock
when the "ring" distributed strategy is enabled.
"""
import math
import torch
from torch import Tensor
from typing import List, Optional, Tuple
from torch.nn.attention.flex_attention import (
      flex_attention,
      AuxRequest,
      create_block_mask,
  )
from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import RingAttentionStrategy

# Use Triton only when block size is big enough (Q_len*K_len)
_TRITON_MIN_WORK = 4096000000000
try:
    from .triton_block import block_softmax_stats_triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
_HAS_TRITON = False


def ring_forward(
    self,
    x,
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None
):
    """LLaMABlock forward pass using ring attention instead of standard attention."""
    residual = x
    x_norm = self.ln(x)

    attn_output = ring_attention(
        x_norm=x_norm,
        attn_module=self.attn,
        strategy=self.distributed_strategy,
        valid_len=self.distributed_strategy._local_valid_len,
        mask=mask,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        causal=is_causal_mask,
    )

    # Unpack attention output
    if use_cache:
        x, cache = attn_output
    else:
        x = attn_output
        cache = None

    x = x + residual

    # then we do FF and Add&Norm
    residual = x
    x = self.ff_ln(x)
    x = self.ff_sub_layer(x)
    x = x + residual

    if use_cache:
        return (x, cache)
    else:
        return x


def ring_attention(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):
    """
    Distributed ring attention with heterogeneous partitioning.
    KV tensors rotate around the ring while Q stays local.
    """
    # decode check
    is_decode = (use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0)

    if is_decode:
        return _ring_attention_pass_q(
            x_norm=x_norm,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=valid_len,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            causal=causal,
        )
    else:
        return _ring_attention_pass_kv(
            x_norm=x_norm,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=valid_len,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            causal=causal,
        )

def _ring_attention_pass_q(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):
    return 0

def _ring_attention_pass_kv(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):
    """
    Ring attention for prefill using pass-KV strategy.
    KV tensors rotate around the ring while Q stays local.
    """
    batch_size, num_valid_tokens_input_shard, emb_dim = x_norm.shape

    assert num_valid_tokens_input_shard == strategy.local_q_len
    current_rank_token_global_start_idx = strategy.local_q_start
    valid_len = strategy.local_q_len

    # slice to valid length to be safe
    current_rank_input_slice = x_norm[:, :valid_len]

    # compute position ids for the current tokens
    if position_ids is not None:
        position_ids_for_rope_computation = position_ids[:, current_rank_token_global_start_idx:current_rank_token_global_start_idx + valid_len]
    elif valid_len > 0:
        position_ids_for_rope_computation = torch.arange(
            current_rank_token_global_start_idx,
            current_rank_token_global_start_idx + valid_len,
            device=x_norm.device
        ).unsqueeze(0).expand(batch_size, -1)
    else:
        position_ids_for_rope_computation = None

    # compute QKV + RoPE for new tokens
    if valid_len:
        q, k, v = _compute_qkv_and_rope(
            attn_module, current_rank_input_slice, position_ids_for_rope_computation
        )
    else:
        nheads, emb_kq_per_head, emb_v_per_head = attn_module.nheads, attn_module.emb_kq_per_head, attn_module.emb_v_per_head
        q = k = torch.empty((batch_size, nheads, 0, emb_kq_per_head), device=x_norm.device, dtype=x_norm.dtype)
        v = torch.empty((batch_size, nheads, 0, emb_v_per_head), device=x_norm.device, dtype=x_norm.dtype)

    scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
    accum_dtype = torch.float32

    # main ring attention with pass-KV
    out = _compute_attention_ring_pass_kv(
        q, k, v, mask, strategy, current_rank_token_global_start_idx, valid_len, scale, accum_dtype, causal
    )

    if valid_len:
        proj = out.transpose(1, 2).reshape(batch_size, valid_len, -1)
        out = attn_module.dense(proj)
    else:
        out = torch.empty((batch_size, 0, emb_dim), device=x_norm.device, dtype=x_norm.dtype)

    # Return cache if requested
    if use_cache:
        return out, (k, v)
    else:
        return out


def _compute_qkv_and_rope(
    attn: MultiHeadAttention,
    x: Tensor,
    rope_position_ids: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, seq_len, _ = x.shape
    q_proj, k_proj, v_proj = attn.in_proj(x, None, None)

    nheads, kvheads = attn.nheads, attn.kvheads
    emb_kq_per_head, emb_v_per_head = attn.emb_kq_per_head, attn.emb_v_per_head

    # reshape & apply RoPE if needed
    q = q_proj.view(batch_size, seq_len, nheads, emb_kq_per_head)
    k = k_proj.view(batch_size, seq_len, kvheads, emb_kq_per_head)
    v = v_proj.view(batch_size, seq_len, kvheads, emb_v_per_head)
    if attn.position_encoder and seq_len:
        assert rope_position_ids is not None
        valid_rope_pos_mask = rope_position_ids.ne(-1)
        if valid_rope_pos_mask.any():
            rope_internal_max_seq_len = getattr(attn.position_encoder, "max_seq_len", 2048)
            clamped_rope_ids = rope_position_ids.clamp(0, rope_internal_max_seq_len - 1)
            q, k = attn.position_encoder.adjusted_qk(q, k, clamped_rope_ids, past_kv_state=None)

    q, k, v = [x_tensor.permute(0, 2, 1, 3) for x_tensor in (q, k, v)]
    if nheads != kvheads:
        kv_to_q_head_ratio = nheads // kvheads
        k = k.repeat_interleave(kv_to_q_head_ratio, dim=1)
        v = v.repeat_interleave(kv_to_q_head_ratio, dim=1)
    return q, k, v


def _online_softmax_update(
    attn_weights: Tensor,
    v_block: Tensor,
    numerator: Tensor,
    denominator: Tensor,
    prev_max_score: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Online softmax update for a single block of attention.
    """
    # Find max in current block
    block_max = attn_weights.max(dim=-1, keepdim=True).values

    # Update global max
    new_max_score = torch.maximum(prev_max_score, block_max)
    # Correction factor for previous accumulations
    correction = torch.exp(prev_max_score - new_max_score)
    # Exp weights for current block (shifted by new max)
    exp_weights = torch.exp(attn_weights - new_max_score)
    # Update numerator and denominator with correction
    numerator = (numerator * correction) + torch.matmul(exp_weights, v_block)
    denominator = (denominator * correction) + exp_weights.sum(dim=-1, keepdim=True)

    return numerator, denominator, new_max_score


def _has_offdiag_contribution(strategy: RingAttentionStrategy, q_start: int, q_len: int, causal: bool) -> bool:
    """
    Check if any off-diagonal block will CONTRIBUTE (not just exist).

    With causal masking, an off-diagonal block is fully masked when:
        k_start > q_end  (all keys are "future" relative to all queries)

    For 2 GPUs:
        - Rank 0: q_end = N/2-1, off-diag k_start = N/2 → k_start > q_end → MASKED
        - Rank 1: q_end = N-1,   off-diag k_start = 0   → k_start ≤ q_end → CONTRIBUTES

    Returns True if merging is needed (can't use Flash Attention shortcut).
    """
    if strategy.world_size == 1:
        return False
    if not causal:
        return True  # All blocks contribute in non-causal

    q_end = q_start + q_len - 1

    # Check each other rank's K block
    for i in range(1, strategy.world_size):
        source_rank = (strategy.rank - i) % strategy.world_size
        k_start = strategy.block_starts[source_rank]
        # If k_start <= q_end, some K positions are not masked → contributes
        if k_start <= q_end:
            return True
    return False


def _compute_attention_ring_pass_kv(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      mask: Optional[Tensor],
      strategy: RingAttentionStrategy,
      q_start: int,
      num_valid_tokens: int,
      scale: float,
      accum_dtype: torch.dtype,
      causal: bool,
  ) -> Tensor:
      B, H, _, Dv = q.shape[0], q.shape[1], q.shape[2], v.shape[-1]

      # Accumulators: normalized output + logsumexp
      out_acc = torch.zeros((B, H, num_valid_tokens, Dv), device=q.device, dtype=accum_dtype)
      lse_acc = torch.full((B, H, num_valid_tokens, 1), float("-inf"), device=q.device, dtype=accum_dtype)

      cur_k, cur_v = k, v
      cur_len = cur_k.shape[2]

      for i in range(strategy.world_size):
          # 1. Start async comm for next iteration
          reqs = None
          if i < strategy.world_size - 1:
              reqs, _ = strategy.ring_shift_kv_async(
                  cur_k, cur_v, cur_len, iteration=i, enable_timing=False
              )

          # 2. Identify source block
          source_rank = (strategy.rank - i) % strategy.world_size
          block_offset = strategy.block_starts[source_rank]

          # 3. Compute attention on current block
          if num_valid_tokens > 0 and cur_len > 0:
              k_start = block_offset
              q_end = q_start + num_valid_tokens - 1
              is_fully_masked = causal and (k_start > q_end)

              if not is_fully_masked:
                  # Build score_mod with global index offsets for causal masking
                  _q_off = q_start
                  _k_off = block_offset

                  if causal:
                      def score_mod(score, b, h, q_idx, kv_idx):
                          return torch.where(
                              q_idx + _q_off >= kv_idx + _k_off,
                              score,
                              torch.tensor(float("-inf")),
                          )
                  else:
                      score_mod = None

                  # flex_attention returns normalized output + lse
                  block_out, aux = flex_attention(
                      q,
                      cur_k[:, :, :cur_len].contiguous(),
                      cur_v[:, :, :cur_len].contiguous(),
                      score_mod=score_mod,
                      scale=1.0 / scale, 
                      return_aux=AuxRequest(lse=True),
                  )

                  # aux.lse shape: [B, H, Q] -> [B, H, Q, 1]
                  block_lse = aux.lse.unsqueeze(-1).to(accum_dtype)
                  block_out = block_out.to(accum_dtype)

                  out_acc, lse_acc = _merge_out_lse(out_acc, lse_acc, block_out, block_lse)

          # 4. Wait for comm
          if i < strategy.world_size - 1:
              assert reqs is not None
              cur_k, cur_v, cur_len, _, sync_event = strategy.ring_shift_kv_wait(
                  reqs, enable_timing=False
              )
              if sync_event is not None:
                  torch.cuda.current_stream().wait_event(sync_event)
              cur_k = cur_k[:, :, :cur_len].contiguous()
              cur_v = cur_v[:, :, :cur_len].contiguous()

      torch.cuda.synchronize()

      if num_valid_tokens == 0:
          return torch.empty((B, H, 0, Dv), device=q.device, dtype=q.dtype)

      return out_acc.to(q.dtype)
def _compute_attention_ring_pass_q():
    return

def _attn_scores(
    Q: Tensor,
    K: Tensor,
    query_indices: Tensor, # global indices for queries in Q
    key_indices: Tensor,   # global indices for keys in K
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tensor:
    batch_size, nheads, num_q, _ = Q.shape
    num_k = K.shape[2]
    if num_q == 0 or num_k == 0:
        return Q.new_empty((batch_size, nheads, num_q, num_k))

    scores = torch.matmul(Q / scale, K.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    if causal:
        # build a [1,1,q_len,k_len] mask where key_pos > query_pos
        future_mask = (key_indices[None, :] > query_indices[:, None])
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(future_mask, float("-inf"))
    return scores


def _merge_out_lse(
      out_acc: Tensor,    # [B, H, Q, Dv]
      lse_acc: Tensor,    # [B, H, Q, 1]
      block_out: Tensor,  # [B, H, Q, Dv]
      block_lse: Tensor,  # [B, H, Q, 1]
  ) -> Tuple[Tensor, Tensor]:
      """Merge two normalized attention outputs using logsumexp."""
      new_lse = torch.logaddexp(lse_acc, block_lse)
      out_acc = (
          torch.exp(lse_acc - new_lse) * out_acc
          + torch.exp(block_lse - new_lse) * block_out
      )
      return out_acc, new_lse
def _online_softmax_merge_stats(
    z_block: Tensor,      # [B, H, Q, D_v]
    l_block: Tensor,      # [B, H, Q, 1]
    m_block: Tensor,      # [B, H, Q, 1]
    numerator: Tensor,    # [B, H, Q, D_v]
    denominator: Tensor,  # [B, H, Q, 1]
    prev_max_score: Tensor,  # [B, H, Q, 1]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Merge a new block's softmax stats (z_block, l_block, m_block)
    into global (numerator, denominator, prev_max_score).
    """
    # new global max per query
    new_max = torch.maximum(prev_max_score, m_block)

    # correction factors
    corr_prev  = torch.exp(prev_max_score - new_max)   # for old accumulators
    corr_block = torch.exp(m_block - new_max)          # for this block

    # merge
    numerator   = numerator * corr_prev  + z_block * corr_block
    denominator = denominator * corr_prev + l_block * corr_block

    return numerator, denominator, new_max

def _block_softmax_stats_naive(
    Q: Tensor,           # [B, H, Q_block, D_k]
    K: Tensor,           # [B, H, K_block, D_k]
    V: Tensor,           # [B, H, K_block, D_v]
    query_indices: Tensor,  # [Q_block] global positions
    key_indices: Tensor,    # [K_block] global positions
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute per-query block stats:
        m_block: max logits in this block
        l_block: sum_j exp(S_ij - m_block_i)
        z_block: sum_j exp(S_ij - m_block_i) * V_j
    using a naive matmul implementation.
    """
    B, H, Q_len, _ = Q.shape
    K_len = K.shape[2]
    Dv = V.shape[-1]

    if Q_len == 0 or K_len == 0:
        m_block = Q.new_full((B, H, Q_len, 1), float("-inf"))
        l_block = Q.new_zeros((B, H, Q_len, 1))
        z_block = Q.new_zeros((B, H, Q_len, Dv))
        return z_block, l_block, m_block

    # 1. logits
    scores = torch.matmul(Q / scale, K.transpose(-2, -1))  # [B, H, Q_len, K_len]

    # 2. apply mask (padding + causal)
    if mask is not None:
        scores = scores + mask.to(scores.dtype)

    if causal:
        # future positions: key_idx > query_idx
        future_mask = (key_indices[None, :] > query_indices[:, None])  # [Q_len, K_len]
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)            # [1,1,Q,K]
        scores = scores.masked_fill(future_mask, float("-inf"))

    # 3. m_block: per-query max
    m_block = scores.max(dim=-1, keepdim=True).values  # [B,H,Q,1]

    # 4. l_block: per-query sumexp
    exp_scores = torch.exp(scores - m_block)           # [B,H,Q,K]
    l_block = exp_scores.sum(dim=-1, keepdim=True)     # [B,H,Q,1]

    # 5. z_block: per-query weighted sum of V
    z_block = torch.matmul(exp_scores, V)              # [B,H,Q,Dv]

    return z_block, l_block, m_block



def _block_softmax_stats(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    query_indices: Tensor,
    key_indices: Tensor,
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Triton path
    if _HAS_TRITON and Q.is_cuda:
        return block_softmax_stats_triton(
            Q, K, V, query_indices, key_indices, scale, mask, causal
        )

    # Fallback: pure PyTorch, correct but slower
    return _block_softmax_stats_naive(
        Q, K, V, query_indices, key_indices, scale, mask, causal
    )
