import torch
from typing import Tuple
from dataclasses import dataclass

from lmcache.cache_engine import LMCacheEngineBuilder
from lmcache.blend.executor import CacheBlendImpl
from lmcache.blend.retriever import SPTBlendRetriever
from lmcache.blend.interfaces import BlendRetrieverTask, BlendExecutor
from lmcache.logging import init_logger

from vllm.attention import AttentionMetadata

from lmcache_vllm.vllm_adapter import ENGINE_NAME

logger = init_logger(__name__)

# TODO: need to load the special token and recompute ratio from configuration
TEMP_SPT = torch.tensor([422], dtype = torch.int, device = "cpu")
RECOMP_RATIO = 0.15
global_blend_retriever = None

@dataclass
class BlendMetadata:
    """The class for cacheblend metadata

    For a single request, there is only one [start, end] tuple in the ROI list
    Tokens:          |-------------------------------------------------|
    Prefix caches:   |*********|
    ROI:                       ^(start)                                ^(end)
    selected tokens: |               *   *      *    *  *    ***       |
    positions:                       ^   ^      ^    ^  ^    ^^^

    When there are multiple requests in a batch, ROI list will have multiple
    start-end tuples

    :ivar int processed_layer_count: The number of processed layers
    :ivar torch.Tensor positions: The positions of the selected tokens
        in the input sequence
    :ivar BlendRetrieverTask retrieval_task: The retrieval task for the
        current request
    :ivar BlendExecutor blend_executor: The blend executor for the current
        request
    """
    processed_layer_count: int
    positions: torch.Tensor
    retrieval_task: BlendRetrieverTask
    blend_executor: BlendExecutor

def pre_initialize():
    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)

    if cache_engine is None:
        logger.error("Cannot initialize cache blend logic because LMCacheEngine is not initialized")
        raise RuntimeError("Cannot initialize cache blend logic because LMCacheEngine is not initialized")

    # FIXME: we are trying to read metadata from cache_engine, which breaks the encapsulation
    global global_blend_retriever 
    global_blend_retriever = SPTBlendRetriever(TEMP_SPT, cache_engine, cache_engine.metadata)

def should_process_request(
        attn_metadata: AttentionMetadata
    ) -> bool:
    has_prefill = attn_metadata.prefill_metadata is not None
    has_decode = attn_metadata.decode_metadata is not None
    if has_prefill and has_decode:
        logger.warning("CacheBlend does not support prefill and decode at the same time")
    return has_prefill and not has_decode

def process_new_request(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> AttentionMetadata:
    """Creates the cacheblend related stuff and put that into the attn metadata
    """
    if not should_process_request(attn_metadata):
        return attn_metadata

    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)

    if cache_engine is None:
        logger.error("Cannot initialize cache blend logic because LMCacheEngine is not initialized")
        raise RuntimeError("Cannot initialize cache blend logic because LMCacheEngine is not initialized")

    global global_blend_retriever
    task = global_blend_retriever.new_request(input_ids, attn_metadata.query_start_loc)
    executor = CacheBlendImpl(RECOMP_RATIO)
    blend_metadata = BlendMetadata(0, positions, task, executor)
    setattr(attn_metadata, "blend_metadata", blend_metadata)
    return attn_metadata


def do_blend(
        fresh_q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata]:
    """Do the cache blending with following steps:
    1. retrieve the KV
    2. blend the KV using executor
    3. updates the blend_metadata
    4. update the attn_metadata for shorter attention

    Returns Q, K, V, and updated attn_metadata
    """
    blend_metadata = getattr(attn_metadata, "blend_metadata", None)
    if blend_metadata is None:
        # Do nothing
        return fresh_q, fresh_k, fresh_v, attn_metadata

    # FIXME: Temp for autocomplete
    blend_metadata = BlendMetadata()

    # Retrieve the KV
    layer_id = blend_metadata.processed_layer_count
    retrieved_kv = blend_metadata.retrieval_task.result(layer_id)

    # blend the KV
    blender_output = blend_metadata.blend_executor.blend(
            layer_id,
            retrieved_kv.k,
            retrieved_kv.v,
            retrieved_kv.valid_mask,
            fresh_q,
            fresh_k,
            fresh_v,
            blend_metadata.positions,
            attn_metadata.query_start_loc,
            0)

    # Update blend_metadata
    attn_metadata.blend_metadata.processed_layer_count += 1
    attn_metadata.blend_metadata.positions = blender_output.positions

    # Update attn_metadata for shorter attention
    if fresh_q.shape != blender_output.q.shape:
        # num_prefills: not change

        # num_prefill_tokens
        attn_metadata.num_prefill_tokens = len(blender_output.positions)

        # slot mapping
        old_slot_mapping = attn_metadata.slot_mapping.clone()
        attn_metadata.slot_mapping = attn_metadata.slot_mapping[blender_output.local_indices]

        # TODO: we should consider changing max_query_len

        # Block tables is for the prefix KV, won't change

        # query_start_loc 
        attn_metadata.query_start_loc = blender_output.query_start_loc

        # context lens: won't change

    return attn_metadata
