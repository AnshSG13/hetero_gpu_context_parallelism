"""
Ring Attention with Heterogeneous GPU Support.

This module implements distributed ring attention for long-context LLM inference,
with support for heterogeneous GPU configurations. Key features:

- Uneven token partitioning across ranks based on GPU capabilities
- Async P2P communication overlapped with compute via separate CUDA streams
- Uses aten._scaled_dot_product_flash_attention for block-wise attention

The main entry point is `ring_attention()`, which is called from LLaMABlock
when the "ring" distributed strategy is enabled.
"""
import math
import torch
from torch import Tensor
from typing import Optional, Tuple

aten = torch.ops.aten
from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import RingAttentionStrategy


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


def _is_causal_behavior(rank: int, world_size: int, source_rank: int, is_causal: bool) -> str:
    """
    Determine causal behavior for a given KV block in ring attention.

    For the diagonal block (source_rank == rank), use is_causal=True.
    For blocks where KV comes from an earlier rank (source_rank < rank),
    all keys are in the past → full attention (no causal mask needed).
    For blocks where KV comes from a later rank (source_rank > rank),
    all keys are in the future → skip entirely.

    Returns: "causal", "full", or "skip"
    """
    if not is_causal:
        return "full"
    if source_rank == rank:
        return "causal"
    elif source_rank < rank:
        return "full"
    else:
        return "skip"


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

          # 3. Compute attention on current block
          if num_valid_tokens > 0 and cur_len > 0:
              is_causal_behavior = _is_causal_behavior(
                  strategy.rank, strategy.world_size, source_rank, causal
              )

              if is_causal_behavior != "skip":
                  k_block = cur_k[:, :, :cur_len].contiguous()
                  v_block = cur_v[:, :, :cur_len].contiguous()

                  # aten._scaled_dot_product_flash_attention returns:
                  # (out, logsumexp, cumulative_seq_len_q, cumulative_seq_len_k,
                  #  max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
                  block_out, block_logsumexp, *_rest = aten._scaled_dot_product_flash_attention(
                      q,
                      k_block,
                      v_block,
                      is_causal=is_causal_behavior == "causal",
                      scale=1.0 / scale,
                  )

                  # block_logsumexp shape: [B, H, Q] -> [B, H, Q, 1]
                  block_lse = block_logsumexp.unsqueeze(-1).to(accum_dtype)
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
