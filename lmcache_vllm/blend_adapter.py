import torch
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from lmcache.cache_engine import LMCacheEngineBuilder
from lmcache.blend.executor import CacheBlendImpl
from lmcache.blend.retriever import SPTBlendRetriever
from lmcache.blend.interfaces import BlendRetrieverTask, BlendExecutor
from lmcache.logging import init_logger

from vllm.attention import AttentionMetadata
from vllm.sequence import SequenceGroupMetadata
from lmcache_vllm.lmcache_utils import ENGINE_NAME

logger = init_logger(__name__)

class ReqId2Indices:
    """The class for cacheblend indices.
    Map request_id to indices to split the prompt.
    """
    def __init__(self):
        self._map_dict = {}
    def add_request(self, request_id, indices):
        self._map_dict[request_id] = indices
    def get_request(self, request_id):
        return self._map_dict.get(request_id, None)
    def delete_request(self, request_id):
        self._map_dict.pop(request_id, None)
    
global_req_id2indices = ReqId2Indices()

# TODO: Special token text and token_id should depend on models.
TEMP_SPT = [422, 422]
global_blend_retriever = None
g_manually_disabled = False

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
    :ivar torch.Tensor selected_token_indices: will be used to update the 
        sampling_metadata after model.forward
    :ivar torch.Tensor original_query_start_loc: The original query start
        loc array before selecting the tokens
    """
    processed_layer_count: int
    positions: torch.Tensor
    retrieval_task: BlendRetrieverTask
    blend_executor: BlendExecutor
    request_prompt_list: List[torch.Tensor]
    prompt_indices_list: List[List[int]]
    selected_token_indices: torch.Tensor
    original_query_start_loc: torch.Tensor

def convert_retrieved_kv_shape(k_or_v: torch.Tensor) -> torch.Tensor:
    """Convert the retrieved KV layer shape to [num_tokens, hidden_dims]
    """
    tmp = k_or_v.squeeze()
    assert tmp.dim() == 3 # Should be [num_tokens, num_heads, head_size]
    nt, nh, hs = tmp.shape
    return tmp.reshape((nt, nh * hs))

def init_cacheblend_retriever():
    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)

    if cache_engine is None:
        logger.error("Cannot initialize cache blend logic because LMCacheEngine is not initialized")
        raise RuntimeError("Cannot initialize cache blend logic because LMCacheEngine is not initialized")

    # FIXME: we are trying to read metadata from cache_engine, which breaks the encapsulation
    global global_blend_retriever 
    global_blend_retriever = SPTBlendRetriever(TEMP_SPT, cache_engine, cache_engine.metadata)



# MAIN FUNCTIONS

def drop_blend_spt(request_id, prompt: List[int]) -> List[int]:
    """Drop the SPT tokens from the prompt and return the new prompt.
    Also stores the indices for later retrieval.
    :param request_id: The request id in sequence group.

    :param prompt: The input prompt after tokenization.
    :type prompt: List[int]

    :return: The new prompt after dropping the SPT tokens.
    :rtype: List[int]
    """
    if global_blend_retriever is None:
        init_cacheblend_retriever()
    new_prompt, indices = global_blend_retriever.drop_spt_and_get_indices(prompt)
    global_req_id2indices.add_request(request_id, indices)
    return new_prompt

def get_blend_indices(request_id) -> List[int]:
    """Get the indices after split for the request id.
    :param request_id: The request id in sequence group.

    :return: The indices for the request id.
    :rtype: List[int]
    """
    indices = global_req_id2indices.get_request(request_id)
    return indices if indices is not None else []

def remove_request_id_indices(request_id):
    """Remove stored indices of request id.
    Called when a sequence group is finished.
    :param request_id: The request id in sequence group.
    """
    global_req_id2indices.delete_request(request_id)

def combine_input_prompt_chunks(
        prompt_chunks: List[str],
    ) -> str:
    """Combine the input chunks by adding the special separators in between

    :param prompt_chunks: A list of input chunks in string
    :type prompt_chunks: List[str]

    :return: The combined input tensor
    :rtype: torch.Tensor
    """
    # TODO: replace hardcoded "<s>" by `tokenizer.special_tokens['bos_token']`
    separator = " # #<s> "
    return separator.join(prompt_chunks)

def append_separator(
        input_prompt: str
    ) -> str:
    """Append the special separator to the input prompt

    :param input_prompt: The input prompt
    :type input_prompt: str

    :return: The input prompt with the special separator appended
    :rtype: str
    """
    separator = " # #"
    return input_prompt + separator

def disable_blend():
    global g_manually_disabled
    g_manually_disabled = True



def should_process_request(
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_caches: List[torch.Tensor],
    ) -> bool:
    # Check if we manually disabled it
    if g_manually_disabled:
        return False

    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    if is_profile_run:
        return False

    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    MINIMUM_TOKENS_TO_ENABLE_BLENDING = cache_engine.config.blend_min_tokens
    if len(input_ids) < MINIMUM_TOKENS_TO_ENABLE_BLENDING:
        return False

    has_prefill = attn_metadata.prefill_metadata is not None
    has_decode = attn_metadata.decode_metadata is not None

    if has_prefill and has_decode:
        logger.warning("CacheBlend does not support prefill and decode at the same time")
    return has_prefill and not has_decode

def process_new_request(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_caches: List[torch.Tensor],
    ) -> AttentionMetadata:
    """Creates the cacheblend related stuff and put that into the attn metadata
    """
    if not should_process_request(input_ids, attn_metadata, kv_caches):
        if hasattr(attn_metadata, "blend_metadata"):
            delattr(attn_metadata, "blend_metadata")
        return attn_metadata
    cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    
    if cache_engine is None:
        logger.error("Cannot initialize cache blend logic because LMCacheEngine is not initialized")
        raise RuntimeError("Cannot initialize cache blend logic because LMCacheEngine is not initialized")

    global global_blend_retriever
    if global_blend_retriever is None:
        init_cacheblend_retriever()
    
    assert hasattr(attn_metadata, "blend_metadata")
    task = global_blend_retriever.new_request(
        attn_metadata.blend_metadata.request_prompt_list, 
        attn_metadata.blend_metadata.prompt_indices_list
    )
    RECOMP_RATIO = cache_engine.config.blend_recompute_ratio
    executor = CacheBlendImpl(RECOMP_RATIO)
    attn_metadata.blend_metadata.positions = positions
    attn_metadata.blend_metadata.retrieval_task = task
    attn_metadata.blend_metadata.blend_executor = executor
    return attn_metadata

def attach_blend_prompt_indices(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        attn_metadata: AttentionMetadata,
    ):
    """Attach the prompts and indices after split to blend_metadata in attn_metadata.
    :param seq_group_metadata_list: The list of sequence group metadata.
    :type seq_group_metadata_list: List[SequenceGroupMetadata]

    :param attn_metadata: The attention metadata.
    :type attn_metadata: AttentionMetadata
    """
    assert not hasattr(attn_metadata, "blend_metadata")
    blend_metadata = BlendMetadata(0, None, None, None, [], [], None, None)
    setattr(attn_metadata, "blend_metadata", blend_metadata)
    seq_lens = attn_metadata.seq_lens
    seq_data_idx = 0
    for seq_group_metadata in seq_group_metadata_list:
        for seqid, seq_data in seq_group_metadata.seq_data.items():
            seq_len = seq_lens[seq_data_idx]
            if seq_group_metadata.block_tables is not None:
                indices = get_blend_indices(seq_group_metadata.request_id)
            else:
                indices = []
            attn_metadata.blend_metadata.request_prompt_list.append(torch.tensor(
                seq_data.get_token_ids()[:seq_len], device="cpu"))
            attn_metadata.blend_metadata.prompt_indices_list.append(indices)
            seq_data_idx += 1
    assert seq_data_idx == len(seq_lens)


def do_blend(
        fresh_q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        attn_metadata: AttentionMetadata,
        rotary_emb,
        reverse_rotary_emb,
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

    # Store original query start loc
    if blend_metadata.original_query_start_loc is None:
        blend_metadata.original_query_start_loc = attn_metadata.query_start_loc.clone()

    # Retrieve the KV
    layer_id = blend_metadata.processed_layer_count
    retrieved_kv = blend_metadata.retrieval_task.result(layer_id)
    if retrieved_kv.k is None or retrieved_kv.v is None:
        # Do nothing if no KV is retrieved
        return fresh_q, fresh_k, fresh_v, attn_metadata

    # blend the KV
    rk = convert_retrieved_kv_shape(retrieved_kv.k)
    rv = convert_retrieved_kv_shape(retrieved_kv.v)
    blend_metadata.blend_executor.set_positional_encoder(rotary_emb)
    blend_metadata.blend_executor.set_reverse_positional_encoder(reverse_rotary_emb)
    blender_output = blend_metadata.blend_executor.blend(
            layer_id,
            rk,
            rv,
            retrieved_kv.valid_mask,
            retrieved_kv.original_positions,
            fresh_q,
            fresh_k,
            fresh_v,
            blend_metadata.positions,
            blend_metadata.original_query_start_loc,
            0)

    # Update blend_metadata
    attn_metadata.blend_metadata.processed_layer_count += 1
    attn_metadata.blend_metadata.positions = blender_output.positions

    # Update attn_metadata for shorter attention
    if fresh_q.shape != blender_output.q.shape:
        # num_prefills: not change

        # num_prefill_tokens 
        attn_metadata.num_prefill_tokens = len(blender_output.positions)

        # slot mapping: don't change slot_mapping because it's for KV

        # TODO: we should consider changing max_query_len

        # Block tables is for the prefix KV, won't change

        # query_start_loc 
        assert blender_output.query_start_loc is not None
        attn_metadata.query_start_loc = blender_output.query_start_loc

        # context lens: won't change

        # selected_token_indices:
        new_selected_token_indices = blender_output.query_start_loc[1:].clone() - 1
        attn_metadata.blend_metadata.selected_token_indices = new_selected_token_indices
    else:
        # Shouldn't change query_start_loc if token selection does not happen
        assert blender_output.query_start_loc is None

    return blender_output.q, blender_output.k, blender_output.v, attn_metadata