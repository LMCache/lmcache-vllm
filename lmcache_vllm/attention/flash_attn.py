import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionType, AttentionLayer
from vllm.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                flash_attn_with_kvcache)
from vllm._custom_ops import reshape_and_cache


def flash_attn_forward_for_cacheblend(
    impl_self: "FlashAttentionImpl",
    layer: "AttentionLayer",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor, 
    kv_cache: torch.Tensor,
    attn_metadata: "FlashAttentionMetadata",
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        output: shape = [num_tokens, num_heads, head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            NOTE: kv_cache will be an empty tensor with shape [0]
            for profiling run.
        attn_metadata: Metadata for attention.
    NOTE: It in-place updates the output tensor.
    """
    # Query handling
    if query.ndim == 3:
        num_tokens = query.shape[0]
        hidden_size = impl_self.num_heads * impl_self.head_size
    elif query.ndim == 2:
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, impl_self.num_heads, impl_self.head_size)
    else:
        raise ValueError(f"Unexpected query dimension: {query.ndim}")

    # Key handling
    if key.ndim != 3:
        if key.ndim == 2:
             key = key.view(num_tokens, impl_self.num_kv_heads, impl_self.head_size)
        else:
             raise ValueError(f"Unexpected key dimension: {key.ndim}")

    # Value handling
    if value.ndim != 3:
        if value.ndim == 2:
             value = value.view(num_tokens, impl_self.num_kv_heads, impl_self.head_size)
        else:
            raise ValueError(f"Unexpected value dimension: {value.ndim}")

    # KV cache handling
    key_cache = None
    value_cache = None
    if kv_cache.numel() > 0:  # Only process if not empty tensor
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Set k_scale and v_scale to 1.0 as tensors
        k_scale_tensor = torch.ones((), device=key_cache.device, dtype=key_cache.dtype)
        v_scale_tensor = torch.ones((), device=value_cache.device, dtype=value_cache.dtype)

        # Call reshape_and_cache with the correctly shaped tensors
        reshape_and_cache(
            key,
            value,
            key_cache.unsqueeze(0),
            value_cache.unsqueeze(0),
            attn_metadata.slot_mapping.flatten(),
            impl_self.kv_cache_dtype,
            k_scale_tensor,
            v_scale_tensor,
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

        if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            prefill_output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.query_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_prefill_seq_len,
                max_seqlen_k=prefill_meta.max_prefill_seq_len,
                softmax_scale=impl_self.scale,
                causal=True,
                window_size=impl_self.sliding_window,
                alibi_slopes=impl_self.alibi_slopes,
                softcap=impl_self.logits_soft_cap,
            )
        else:
            # prefix-enabled attention
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output = flash_attn_varlen_func(  # noqa
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_k=max_seq_len,
                softmax_scale=impl_self.scale,
                causal=True,
                alibi_slopes=impl_self.alibi_slopes,
                block_table=prefill_meta.block_tables,
                softcap=impl_self.logits_soft_cap,
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
        if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            prefill_output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.query_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_prefill_seq_len,
                max_seqlen_k=prefill_meta.max_prefill_seq_len,
                softmax_scale=impl_self.scale,
                causal=True,
                window_size=impl_self.sliding_window,
                alibi_slopes=impl_self.alibi_slopes,
                softcap=impl_self.logits_soft_cap,
            )
        else:
            # prefix-enabled attention
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output = flash_attn_varlen_func(  # noqa
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_k=max_seq_len,
                softmax_scale=impl_self.scale,
                causal=True,
                alibi_slopes=impl_self.alibi_slopes,
                block_table=prefill_meta.block_tables,
                softcap=impl_self.logits_soft_cap,
            )

    if decode_meta := attn_metadata.decode_metadata:
        # Only do decoding if we have a valid cache
        if kv_cache.numel() > 0:
            # Decoding run.
            decode_output = flash_attn_with_kvcache(
                decode_query.unsqueeze(1),
                key_cache,
                value_cache,
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=impl_self.scale,
                causal=True,
                alibi_slopes=impl_self.alibi_slopes,
                softcap=impl_self.logits_soft_cap,
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
    # Ensure the original forward exists before patching
    if hasattr(vllm.attention.backends.flash_attn.FlashAttentionImpl, 'forward'):
        vllm.attention.backends.flash_attn.FlashAttentionImpl.forward = flash_attn_forward_for_cacheblend
    else:
        print("Warning: vllm.attention.backends.flash_attn.FlashAttentionImpl.forward not found for patching.")
