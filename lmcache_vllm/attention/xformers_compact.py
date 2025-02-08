"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)

from vllm.attention.backends.xformers import _get_seq_len_block_table_args
import lmc_ops
from lmcache.logging import init_logger
from lmcache.compactor import LMCacheCompactorBuilder
from lmcache_vllm.attention.prefix_prefill_compact import forward_prefix_expose
import os

logger = init_logger(__name__)


def xformers_forward_compact(
    self,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    kv_cache: Optional[torch.Tensor],
    attn_metadata: "XFormersMetadata",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    attn_type: AttentionType = AttentionType.DECODER,
) -> torch.Tensor:
    """Forward pass with xFormers and PagedAttention.

    For decoder-only models: query, key and value must be non-None.

    For encoder/decoder models:
    * XFormersImpl.forward() may be invoked for both self- and cross-
        attention layers.
    * For self-attention: query, key and value must be non-None.
    * For cross-attention:
        * Query must be non-None
        * During prefill, key and value must be non-None; key and value
            get cached for use during decode.
        * During decode, key and value may be None, since:
            (1) key and value tensors were cached during prefill, and
            (2) cross-attention key and value tensors do not grow during
                decode
    
    A note on how the attn_type (attention type enum) argument impacts
    attention forward() behavior:

        * DECODER: normal decoder-only behavior;
            use decoder self-attention block table
        * ENCODER: no KV caching; pass encoder sequence
            attributes (encoder_seq_lens/encoder_seq_lens_tensor/
            max_encoder_seq_len) to kernel, in lieu of decoder
            sequence attributes (seq_lens/seq_lens_tensor/max_seq_len)
        * ENCODER_DECODER: cross-attention behavior;
            use cross-attention block table for caching KVs derived
            from encoder hidden states; since KV sequence lengths
            will match encoder sequence lengths, pass encoder sequence
            attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
            max_encoder_seq_len)

    Args:
        query: shape = [num_tokens, num_heads * head_size]
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
        attn_metadata: Metadata for attention.
        attn_type: Select attention type, between encoder attention,
                    decoder self-attention, or encoder/decoder cross-
                    attention. Defaults to decoder self-attention,
                    which is the vLLM default generally
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """

    # Check that appropriate attention metadata attributes are
    # selected for the desired attention type
    if (attn_type == AttentionType.ENCODER
            and (not attn_metadata.is_all_encoder_attn_metadata_set)):
        raise AttributeError("Encoder attention requires setting "
                                "encoder metadata attributes.")
    elif (attn_type == AttentionType.ENCODER_DECODER
            and (not attn_metadata.is_all_cross_attn_metadata_set)):
        raise AttributeError("Encoder/decoder cross-attention "
                                "requires setting cross-attention "
                                "metadata attributes.")

    query = query.view(-1, self.num_heads, self.head_size)
    if key is not None:
        assert value is not None
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
    else:
        assert value is None

    # Self-attention vs. cross-attention will impact
    # which KV cache memory-mapping & which
    # seqlen datastructures we utilize

    if (attn_type != AttentionType.ENCODER and kv_cache is not None):
        # KV-cache during decoder-self- or
        # encoder-decoder-cross-attention, but not
        # during encoder attention.
        #
        # Even if there are no new key/value pairs to cache,
        # we still need to break out key_cache and value_cache
        # i.e. for later use by paged attention
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        if (key is not None) and (value is not None):

            if attn_type == AttentionType.ENCODER_DECODER:
                # Update cross-attention KV cache (prefill-only)
                # During cross-attention decode, key & value will be None,
                # preventing this IF-statement branch from running
                updated_slot_mapping = attn_metadata.cross_slot_mapping
            else:
                # Update self-attention KV cache (prefill/decode)
                updated_slot_mapping = attn_metadata.slot_mapping

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory
            # profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                updated_slot_mapping,
                                                self.kv_cache_dtype,
                                                k_scale, v_scale)

    if attn_type != AttentionType.ENCODER:
        # Decoder self-attention supports chunked prefill.
        # Encoder/decoder cross-attention requires no chunked
        # prefill (100% prefill or 100% decode tokens, no mix)
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
    else:
        # Encoder attention - chunked prefill is not applicable;
        # derive token-count from query shape & and treat them
        # as 100% prefill tokens
        assert attn_metadata.num_encoder_tokens is not None
        num_prefill_tokens = attn_metadata.num_encoder_tokens
        num_decode_tokens = 0

    if attn_type == AttentionType.DECODER:
        # Only enforce this shape-constraint for decoder
        # self-attention
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

    output = torch.empty_like(query)
    # Query for decode. KV is not needed because it is already cached.
    decode_query = query[num_prefill_tokens:]
    # QKV for prefill.
    query = query[:num_prefill_tokens]
    if key is not None and value is not None:
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

    assert query.shape[0] == num_prefill_tokens
    assert decode_query.shape[0] == num_decode_tokens

    if prefill_meta := attn_metadata.prefill_metadata:
        # Prompt run.
        if kv_cache is None or prefill_meta.block_tables.numel() == 0:
            logger.info(f"prefill using normal attention {prefill_meta.block_tables.numel()}")
            # normal attention.
            # block tables are empty if the prompt does not have a cached
            # prefix.
            out = self._run_memory_efficient_xformers_forward(
                query, key, value, prefill_meta, attn_type=attn_type)
            assert out.shape == output[:num_prefill_tokens].shape
            output[:num_prefill_tokens] = out
        else:
            logger.info(f"prefill using paged attention {prefill_meta.block_tables.numel()}")

            assert prefill_meta.query_start_loc is not None
            assert prefill_meta.max_query_len is not None

            # prefix-enabled attention
            # TODO(Hai) this triton kernel has regression issue (broke) to
            # deal with different data types between KV and FP8 KV cache,
            # to be addressed separately.
            # logger.info('------------------------prefill_meta:')
            # for k, v in prefill_meta.__dict__.items():
            #     logger.info(f"\t{k}: {v}")
            out, raw_qk = forward_prefix_expose(
                query,
                key,
                value,
                self.kv_cache_dtype,
                key_cache,
                value_cache,
                prefill_meta.block_tables,
                prefill_meta.query_start_loc,
                prefill_meta.seq_lens_tensor,
                prefill_meta.context_lens_tensor,
                prefill_meta.max_query_len,
                self.alibi_slopes,
                self.sliding_window,
                k_scale,
                v_scale,
            )
            print('=layer=')
            torch.set_printoptions(profile='full')
            # for head in probabilities[0]:
            #     print(head[13])
            print(raw_qk[0][10])
            print(raw_qk[0][10].shape)
            print(raw_qk.shape)

            # out = PagedAttention.forward_prefix(
            #     query,
            #     key,
            #     value,
            #     self.kv_cache_dtype,
            #     key_cache,
            #     value_cache,
            #     prefill_meta.block_tables,
            #     prefill_meta.query_start_loc,
            #     prefill_meta.seq_lens_tensor,
            #     prefill_meta.context_lens_tensor,
            #     prefill_meta.max_query_len,
            #     self.alibi_slopes,
            #     self.sliding_window,
            #     k_scale,
            #     v_scale,
            # )
            assert output[:num_prefill_tokens].shape == out.shape
            output[:num_prefill_tokens] = out

    if decode_meta := attn_metadata.decode_metadata:

        (
            seq_lens_arg,
            max_seq_len_arg,
            block_tables_arg,
        ) = _get_seq_len_block_table_args(decode_meta, False, attn_type)

        # NOTE(Jiayi): Modification starts
        # FIXME(Jiayi): need an if loop to enable lmcache_compactor
        # FIXME(Jiayi): Fix the arguments below
        if os.getenv("LMC_COMPACTOR", None) == "True":
            total_seq_len = torch.sum(seq_lens_arg)
            lmcache_compactor = LMCacheCompactorBuilder.get(instance_id="lmcache_compactor")
            logits_buffer_queue = lmcache_compactor.logits_buffer_queue
            
            logits_store = logits_buffer_queue.get()
            output_compact = torch.empty_like(decode_query)
            block_size = value_cache.shape[3]
            
            # logger.info('decode using compact attention')
            lmc_ops.paged_attention_compact_v1(
                logits_store,
                output_compact,
                decode_query,
                key_cache,
                value_cache,
                self.num_kv_heads,
                self.scale,
                block_tables_arg,
                seq_lens_arg,
                block_size,
                max_seq_len_arg,
                self.alibi_slopes,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
                0,
                0,
                0,
                64,
                0,
            )
            output[num_prefill_tokens:] = output_compact
            
            # update buffers here
            logits_buffer_queue.put(logits_store)
            
        
        else:
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )
        # NOTE(Jiayi): Modification ends

    # Reshape the output tensor.
    return output.view(-1, self.num_heads * self.head_size)

def inject_xformers_compact():
    import vllm.attention.backends.xformers
    vllm.attention.backends.xformers.XFormersImpl.forward = xformers_forward_compact