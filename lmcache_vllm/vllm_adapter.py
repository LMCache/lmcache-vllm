from typing import TYPE_CHECKING, List, Optional, Tuple
from enum import Enum
import os
import torch
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm import _custom_ops as ops
from vllm.sequence import SequenceGroupMetadata
from vllm.config import ModelConfig, ParallelConfig, CacheConfig

from lmcache.logging import init_logger
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.utils import _lmcache_nvtx_annotate


logger = init_logger(__name__)

ENGINE_NAME = "vllm-instance"
LMCACHE_CUDA_STREAM = torch.cuda.Stream()

class StoreStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2
    DECODE = 3
    NONE = 4

class RetrieveStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2 # not last chunk
    CHUNK_PREFILL_LAST = 3
    NONE = 4


def init_lmcache_engine(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine by the given model config and parallel 
    config. This function will check the environment variable 
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param model_config: The model configuration in vLLM.
    :type model_config: ModelConfig
    :param parallel_config: The parallel configuration in vLLM.
    :type parallel_config: ParallelConfig

    :return: The initialized LMCache engine or None (if the environment variable
        `LMCACHE_CONFIG_FILE` is not set).
    :rtype: Optional[LMCacheEngine]
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return 


    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn("No LMCache configuration file is set. Returning default config")
        logger.warn("Please set the configuration file through "
                    "the environment variable: LMCACHE_CONFIG_FILE")
        config = LMCacheEngineConfig.from_defaults(
                local_device = "cpu",
                remote_url = None,
                remote_serde = None,
                pipelined_backend = False)
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)
    
    metadata = LMCacheEngineMetadata(
            model_config.model,
            parallel_config.world_size,
            parallel_config.rank,
            "vllm")
    
    engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            config,
            metadata)

    return engine

def close_lmcache_engine() -> None:
    """Close the LMCache engine if it is initialized.
    """
    logger.debug("Closing LMCache Engine")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

def lmcache_should_retrieve(
        model_input: "ModelInputForGPUWithSamplingMetadata", 
        kv_caches: List[torch.Tensor]) -> RetrieveStatus:
    """Check should we retrieve KV from LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: RetrieveStatus.
    """

    seq_lens = model_input.attn_metadata.seq_lens
    
    has_engine = LMCacheEngineBuilder.get(ENGINE_NAME) is not None
    if not has_engine or kv_caches is None:
        return RetrieveStatus.NONE

    attn_meta = model_input.attn_metadata
    prefill_meta = attn_meta.prefill_metadata
    
    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    if is_profile_run:
        return RetrieveStatus.NONE
    
    # check if the current run is prefill
    # TODO (Jiayi): chunked prefill + prefix caching in a single batch
    # is not and should not be supported here
    # what about multuple chunk prefills in a single batch??
    
    # Assume all chunks are prefills
    is_all_prefill_run = ((attn_meta.num_prefills == len(model_input.seq_lens))\
        and prefill_meta is not None)
    if is_all_prefill_run:
        selected_token_indices = model_input.sampling_metadata.selected_token_indices
        if len(selected_token_indices) == 0:
            # There should only be 1 chunk in chunked prefill
            assert len(model_input.seq_lens) == 1
            return RetrieveStatus.CHUNK_PREFILL
        
        '''
        # Check whether the current prefill chunk is the last one
        key = list(model_input.sampling_metadata.seq_groups[0].seq_data.keys())[0]
        seq_data = model_input.sampling_metadata.seq_groups[0].seq_data[key]
        prompt_tokens = seq_data.prompt_token_ids
        if model_input.seq_lens[0] != len(prompt_tokens):
            #TODO(Jiayi): We should support this as well
            return RetrieveStatus.CHUNK_PREFILL_LAST
        '''
        return RetrieveStatus.PREFILL

    # for disaggregated prefilling: allow bypassing model execution

    return RetrieveStatus.NONE


def lmcache_should_store(
        model_input: "ModelInputForGPUWithSamplingMetadata", 
        kv_caches: List[torch.Tensor]) -> StoreStatus:
    """Check should we store KV into LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: A list of StoreStatus.
             StoreStatus.PREFILL/DECODE if we should store KV after PREFILL/DECODE.
             StoreStatus.NONE if no storing is required.
    """
    seq_lens = model_input.attn_metadata.seq_lens
    store_status = [StoreStatus.NONE] * len(seq_lens)
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    has_engine = engine is not None
    if not has_engine:
        return store_status


    attn_meta = model_input.attn_metadata
    prefill_meta = attn_meta.prefill_metadata
    
    # TODO (yihua): Current implementation is in GPUModelRunner, so we do
    #               not need to check the type of model_runner
    #from vllm.worker.model_runner import GPUModelRunnerBase
    #if not isinstance(model_runner, GPUModelRunnerBase):
    #    return False

    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    
    if is_profile_run:
        return store_status

    # FIXME(Jiayi): need to support chunked prefill (batch prefill and decode)
    # check if the current run is prefill
    is_all_prefill_run = ((attn_meta.num_prefills == len(model_input.seq_lens))\
        and (prefill_meta is not None))

    if is_all_prefill_run:
        selected_token_indices = model_input.sampling_metadata.selected_token_indices
        if len(selected_token_indices) == 0:
            # There should only be 1 chunk in chunked prefill
            assert len(seq_lens) == 1
            return [StoreStatus.CHUNK_PREFILL]
        key = list(model_input.sampling_metadata.seq_groups[0].seq_data.keys())[0]
        seq_data = model_input.sampling_metadata.seq_groups[0].seq_data[key]
        prompt_tokens = seq_data.prompt_token_ids
        
        if len(prompt_tokens)-1 != selected_token_indices[0]:
            # last chunk in chunk prefill
            assert len(seq_lens) == 1
            return [StoreStatus.NONE]
        return [StoreStatus.PREFILL] * len(seq_lens)
        

    # Determine whether to save decoded KV cache
    #seq_groups = model_input.sampling_metadata.seq_groups
    if engine.save_decode_cache:
        seq_lens = model_input.attn_metadata.seq_lens

        for idx, seq_len in enumerate(seq_lens):
            if seq_len % engine.chunk_size == 0:
                store_status[idx] = StoreStatus.DECODE
    return store_status


@_lmcache_nvtx_annotate
def lmcache_store_kv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    cache_config: CacheConfig,
    kv_caches: List[torch.Tensor],
    store_status: List[StoreStatus],
) -> None:
    """Store the KV caches into LMCache for the current model_input.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to get KV from
    :type kv_caches: List[torch.Tensor]
    
    :param store_status: Indicate whether and how KV cache of each req is stored
    :type store_status: List[StoreStatus]
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    seq_lens = model_input.attn_metadata.seq_lens
        
    if hasattr(model_executable.model, "start_layer"):
        start_layer = model_executable.model.start_layer
    else:
        start_layer = 0

    if hasattr(model_executable.model, "end_layer"):
        end_layer = model_executable.model.end_layer
    else:
        end_layer = len(kv_caches)

    seq_data_idx = 0
    seq_group_metadata_list = model_input.seq_group_metadata_list
    for seq_group_metadata in seq_group_metadata_list:
        for seqid, seq_data in seq_group_metadata.seq_data.items():
            status = store_status[seq_data_idx]
            if status == StoreStatus.NONE:
                continue
            elif status == StoreStatus.DECODE:
                if seq_len % engine.chunk_size != 0:
                    continue
            # TODO (Jiayi): can chunk prefill and vllm prefix caching use the same logic?
            if status == StoreStatus.CHUNK_PREFILL:
                seq_len = seq_lens[seq_data_idx]
            else:
                seq_len = seq_data.get_len()
            
            current_tokens = torch.tensor(seq_data.get_token_ids()[:seq_len], device="cpu")
            vllm_block_size = cache_config.block_size
            kv_tensors_mask = ~engine.lookup(current_tokens, True)
            from vllm.attention.backends.utils import compute_slot_mapping
            slot_mapping = []
            compute_slot_mapping(False, slot_mapping, seqid, seq_len, 
                0, 0, vllm_block_size, seq_group_metadata.block_tables)
            current_slot_mapping_tensor = torch.tensor(slot_mapping, device="cpu")
            current_slot_mapping_tensor = current_slot_mapping_tensor[kv_tensors_mask]
            kv_tuple_list = []
            if len(current_slot_mapping_tensor) > 0:
                for layer_id in range(start_layer, end_layer):
                    kv_cache = kv_caches[layer_id - start_layer]

                    _, _, num_heads, head_size = kv_cache[0].shape

                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
                    
                    kv_tuple_list.append(
                            (key_cache[current_slot_mapping_tensor],
                            value_cache[current_slot_mapping_tensor])
                        )

                stored_token_num = len(current_slot_mapping_tensor)
                skipped_token_num = seq_len - stored_token_num
                logger.debug(f"Store skips {skipped_token_num} tokens and then stores {stored_token_num} tokens")
                engine.store(current_tokens.cpu(), tuple(kv_tuple_list), kv_tensors_mask,
                            skip_existing = True, blocking = False)
            
            seq_data_idx += 1

@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    model_executable,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    retrieve_status: RetrieveStatus,
) -> Tuple["ModelInputForGPUWithSamplingMetadata", bool]:
    """Retrieve the KV caches from LMCache for the current model_input. And 
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :param retrieve_status: Indicate whether and how KV cache of each req is retrieved
    :type retrieve_status: List[RetrieveStatus]
    
    :return: The rebuilt model_input to reflect the changes in KV.
    :return: The boolean value to indicate whether the entire execute_model should be skipped
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    
    # This is disagg decode instance, during prefill state
    # Need to receive KV from the prefill instance
    query_start_loc = model_input.attn_metadata.query_start_loc
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    seq_lens = model_input.attn_metadata.seq_lens

    if hasattr(model_executable.model, "start_layer"):
        start_layer = model_executable.model.start_layer
    else:
        start_layer = 0

    if hasattr(model_executable.model, "end_layer"):
        end_layer = model_executable.model.end_layer
    else:
        end_layer = len(kv_caches)
    
    # The following metadata are needed to rebuilt the model input
    full_tokens_list = []
    num_computed_tokens_list = []
    lmc_num_computed_tokens_list= []
    
    start_pos_list = []
    is_prefill_list = []
    seq_group_metadata_list = model_input.seq_group_metadata_list
    next_start_pos = 0
    num_request_not_found = 0
    temp_block_table_list = []
    
    # idx is on a sequence, not a sequence group.
    idx = 0

    for seq_group_metadata in seq_group_metadata_list:
        request_id = seq_group_metadata.request_id
        seq_ids = model_input.request_ids_to_seq_ids[request_id]
        for seq_id in seq_ids:
            seq_data = seq_group_metadata.seq_data[seq_id]
            

            if retrieve_status == RetrieveStatus.CHUNK_PREFILL:
                total_seq_len = seq_lens[idx]
            else:
                total_seq_len = seq_data.get_len()
            full_token_tensor = torch.tensor(seq_data.get_token_ids()[:total_seq_len], device="cpu")
            full_tokens_list.append(full_token_tensor)
            
            vllm_num_required_tokens = (query_start_loc[idx + 1] - query_start_loc[idx]).item()
            
            start_pos = next_start_pos
            end_pos = start_pos + vllm_num_required_tokens
            next_start_pos = end_pos
            start_pos_list.append(start_pos)
            
            # TODO(Jiayi): is deepcopy avoidable here?
            temp_block_table = deepcopy(seq_group_metadata.block_tables[seq_id])
            temp_block_table_list.append(temp_block_table)
                
            # number of tokens computed by vllm (e.g., chunk prefill, prefix caching)
            vllm_num_computed_tokens = total_seq_len - vllm_num_required_tokens
            
            # construct token mesk to indicate what tokens should be retrieved
            # from lmc. Tokens computed in vllm already shoudl be skipped
            token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
            token_mask[:vllm_num_computed_tokens] = False
            
            # call lmcache retrieve
            kv_tuple, ret_token_mask = engine.retrieve(full_token_tensor, token_mask)
            lmc_num_computed_tokens = torch.sum(ret_token_mask).item()
            
            # total number of computed tokens (vllm + lmc)
            num_computed_tokens = vllm_num_computed_tokens + lmc_num_computed_tokens
            
            # TODO(Jiayi): currently we do not skip anything if chunked prefill
            # is batched with any decode or other chunked prefills.
            # This is not done as I assume the performance benefit is marginal.
            if retrieve_status == RetrieveStatus.CHUNK_PREFILL:
                if num_computed_tokens != vllm_num_required_tokens:
                    return model_input, False
            else:
                # Avoid the error when prefix is exactly the same as the retrieved
                # However, in chunk prefill, the entire prefill should be skipped
                if num_computed_tokens == total_seq_len:
                    num_computed_tokens -= 1
            
            num_computed_tokens_list.append(num_computed_tokens)
            lmc_num_computed_tokens_list.append(lmc_num_computed_tokens)
            
            
            # No cache found, move on
            if lmc_num_computed_tokens == 0:
                num_request_not_found += 1
                idx += 1
                continue
            
            
            # Inject the lmc retrieved kv cache
            logger.debug(f"Injected token number: {lmc_num_computed_tokens}")
            for i in range(start_layer, end_layer):
                layer_idx = i - start_layer
                kv_cache = kv_caches[layer_idx]
                layer = model_executable.model.layers[i]
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    kv_tuple[layer_idx][0].to(key_cache.device),
                    kv_tuple[layer_idx][1].to(value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:start_pos + lmc_num_computed_tokens],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )
            
            idx += 1
            
            # TODO(Jiayi): `is_prefill` seems redundant
            # please try to remove this in the future 
            is_prefill_list.append(True)
    
    seq_cnt = len(query_start_loc) - 1
    assert idx == seq_cnt
    assert len(lmc_num_computed_tokens_list) == seq_cnt
    assert len(num_computed_tokens_list) == seq_cnt
    
    if retrieve_status == RetrieveStatus.CHUNK_PREFILL and \
        num_request_not_found == 0:
        return model_input, True  
            
    # Some of the request can be skipped for a bit
    # TODO(Jiayi): need e2e test full prefill and partial prefill
    # in a single batch
    if num_request_not_found < seq_cnt:
        rebuilt_model_input = build_partial_prefill_input(
            model_input,
            full_tokens_list,
            num_computed_tokens_list,
            start_pos_list,
            slot_mapping,
            lmc_num_computed_tokens_list,
            is_prefill_list,
            seq_group_metadata_list,
            temp_block_table_list,
            device=kv_cache[0].device,
        )
        logger.debug("Rebuilt the input!")
        return rebuilt_model_input, False
    
    logger.debug("Returning the original input!")
    return model_input, False

def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    full_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    lmc_num_computed_tokens_list: List[int],
    is_prefill_list: List[bool],
    seq_group_metadata_list: List[SequenceGroupMetadata],
    temp_block_table_list: List[List[int]],
    device: torch.device,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Helper function to rebuild the model input for the current request.
    """
    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []
    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    last_query_start_loc = 0

    # recounting query and context lengths
    for idx in range(len(full_tokens_list)):
        token_tensor = full_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        start_pos = start_pos_list[idx]
        is_prefill = is_prefill_list[idx]
        lmc_num_computed_tokens = lmc_num_computed_tokens_list[idx]
        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        q_len = num_token - num_computed_token
        assert q_len > 0
        rebuilt_query_lens.append(q_len)
        start_input_pos_idx = start_pos + lmc_num_computed_tokens
        end_input_pos_idx = start_input_pos_idx + q_len
        rebuilt_input_positions.append(
            model_input.input_positions[start_input_pos_idx: end_input_pos_idx])
        # Attn metadata-related
        if is_prefill:
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
        else:
            assert q_len == 1
        
        start_slot_idx = start_pos + lmc_num_computed_tokens
        end_slot_idx = start_slot_idx + q_len
        new_slot_mapping = slot_mapping_flat[start_slot_idx:end_slot_idx]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
        temp_block_table = temp_block_table_list[idx]
        temp_block_table_tensor = torch.tensor(temp_block_table).to(model_input.attn_metadata.block_tables.dtype)
        rebuilt_block_tables.append(temp_block_table_tensor)
        last_query_start_loc += q_len
        rebuilt_query_start_loc.append(last_query_start_loc)  # start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # Sampling metadata related
        # seq_groups (use rebuilt query lens)
        rebuilt_selected_token_indices.append(last_query_start_loc - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(
        rebuilt_slot_mapping).to(device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = pad_sequence(
        rebuilt_block_tables,
        batch_first=True
        ).to(device)
    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None
    rebuilt_sampling_metadata = None
    # rebuilt sampling_metadata
    if model_input.sampling_metadata is not None:
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

    # import here to avoid circular import.
    from vllm.worker.model_runner import (
        ModelInputForGPUWithSamplingMetadata)
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
        async_callback=model_input.async_callback,
        seq_group_metadata_list=seq_group_metadata_list
    )

    return rebuilt_model_input

