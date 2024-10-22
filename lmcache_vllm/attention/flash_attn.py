import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionType


def flash_attn_forward_for_cacheblend(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: "FlashAttentionMetadata",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    attn_type: AttentionType = AttentionType.DECODER,
) -> torch.Tensor:
    """Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads * head_size]
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """
    if attn_type != AttentionType.DECODER:
        raise NotImplementedError("Encoder self-attention and "
                                  "encoder/decoder cross-attention "
                                  "are not implemented for "
                                  "FlashAttentionImpl")

    # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
    assert k_scale == 1.0 and v_scale == 1.0, (
        "key/v_scale is not supported in FlashAttention.")

    num_tokens, hidden_size = query.shape
    # Reshape the query, key, and value tensors.
    query = query.view(-1, self.num_heads, self.head_size)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size)

    if kv_cache is not None:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Reshape the input keys and values and store them in the cache.
        # If kv_cache is not provided, the new key and value tensors are
        # not cached. This happens during the initial memory profiling run.
        torch.ops.vllm.reshape_and_cache_flash(
            key,
            value,
            kv_cache,
            attn_metadata.slot_mapping.flatten(),
            self.kv_cache_dtype,
            k_scale,
            v_scale,
        )

    num_prefill_tokens = attn_metadata.num_prefill_tokens
    num_decode_tokens = attn_metadata.num_decode_tokens

    # Injection for CacheBlend
    if key.shape[0] > query.shape[0]:
        # Cache blend forward
        num_kv_tokens = key.shape[0]
        assert value.shape[0] == num_kv_tokens
        assert query.shape[0] == num_prefill_tokens

        # In the cacheblend case, prefill_meta must be not None
        prefill_meta = attn_metadata
        assert prefill_meta is not None

        if (kv_cache is None or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            prefill_output = torch.ops.vllm.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.query_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_prefill_seq_len,
                max_seqlen_k=prefill_meta.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            )
        else:
            # prefix-enabled attention
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output = torch.ops.vllm.flash_attn_varlen_func(  # noqa
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                block_table=prefill_meta.block_tables,
                softcap=self.logits_soft_cap,
            )

        assert prefill_output is not None
        return prefill_output.view(num_prefill_tokens, hidden_size)
    # End of injection


    assert key.shape[0] == num_prefill_tokens + num_decode_tokens
    assert value.shape[0] == num_prefill_tokens + num_decode_tokens

    # Query for decode. KV is not needed because it is already cached.
    decode_query = query[num_prefill_tokens:]
    # QKV for prefill.
    query = query[:num_prefill_tokens]
    key = key[:num_prefill_tokens]
    value = value[:num_prefill_tokens]

    assert query.shape[0] == num_prefill_tokens
    assert decode_query.shape[0] == num_decode_tokens

    prefill_output: Optional[torch.Tensor] = None
    decode_output: Optional[torch.Tensor] = None

    if prefill_meta := attn_metadata.prefill_metadata:
        # Prompt run.
        if (kv_cache is None or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            prefill_output = torch.ops.vllm.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.seq_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_prefill_seq_len,
                max_seqlen_k=prefill_meta.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            )
        else:
            # prefix-enabled attention
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output = torch.ops.vllm.flash_attn_varlen_func(  # noqa
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                block_table=prefill_meta.block_tables,
                softcap=self.logits_soft_cap,
            )

    if decode_meta := attn_metadata.decode_metadata:
        # Decoding run.
        decode_output = torch.ops.vllm.flash_attn_with_kvcache(
            decode_query.unsqueeze(1),
            key_cache,
            value_cache,
            block_table=decode_meta.block_tables,
            cache_seqlens=decode_meta.seq_lens_tensor,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            softcap=self.logits_soft_cap,
        ).squeeze(1)

    if prefill_output is None:
        assert decode_output is not None
        return decode_output.view(num_decode_tokens, hidden_size)
    if decode_output is None:
        assert prefill_output is not None
        return prefill_output.view(num_prefill_tokens, hidden_size)
    output = torch.cat([prefill_output, decode_output], dim=0)
    return output.view(num_tokens, hidden_size)

def inject_flash_attn():
    import vllm.attention.backends.flash_attn
    vllm.attention.backends.flash_attn.FlashAttentionImpl.forward = flash_attn_forward_for_cacheblend
