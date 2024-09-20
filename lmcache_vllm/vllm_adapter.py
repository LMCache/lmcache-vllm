from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import os
import torch
from copy import deepcopy

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm import _custom_ops as ops
from vllm.sequence import IntermediateTensors
from vllm.config import ModelConfig, ParallelConfig

from lmcache.logging import init_logger
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata


logger = init_logger(__name__)

ENGINE_NAME = "vllm-instance"

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
        logger.warn("No LMCache configuration file is set. Returning None")
        logger.warn("Please set the configuration file through "
                    "the environment variable: LMCACHE_CONFIG_FILE")
        return None

    config_file = os.environ["LMCACHE_CONFIG_FILE"]
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
        kv_caches: List[torch.Tensor]) -> bool:
    """Check should we retrieve KV from LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: True if we should retrieve KV from LMCache, False otherwise.
    """

    has_engine = LMCacheEngineBuilder.get(ENGINE_NAME) is not None
    if not has_engine or kv_caches is None:
        return False

    prefill_meta = model_input.attn_metadata.prefill_metadata

    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    # check if the current run is prefill
    is_prefill_run = prefill_meta is not None
    # for disaggregated prefilling: allow bypassing model execution

    return all([
        is_prefill_run, not is_profile_run
    ])


def lmcache_should_store(
        model_input: "ModelInputForGPUWithSamplingMetadata", 
        kv_caches: List[torch.Tensor]) -> bool:
    """Check should we store KV into LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: True if we should store KV into LMCache, False otherwise.
    """
    has_engine = LMCacheEngineBuilder.get(ENGINE_NAME) is not None
    if not has_engine or kv_caches is None:
        return False

    prefill_meta = model_input.attn_metadata.prefill_metadata

    # TODO (yihua): Current implementation is in GPUModelRunner, so we do
    #               not need to check the type of model_runner
    #from vllm.worker.model_runner import GPUModelRunnerBase
    #if not isinstance(model_runner, GPUModelRunnerBase):
    #    return False

    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    # check if the current run is prefill
    is_prefill_run = prefill_meta is not None


    return all([
        is_prefill_run, not is_profile_run, 
    ])



def lmcache_store_kv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor]
) -> None:
    """Store the KV caches into LMCache for the current model_input.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to get KV from
    :type kv_caches: List[torch.Tensor]
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    input_tokens_tensor = model_input.input_tokens
    seq_lens = model_input.attn_metadata.seq_lens
    slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
    if hasattr(model_executable.model, "start_layer"):
        start_layer = model_executable.model.start_layer
    else:
        start_layer = 0

    if hasattr(model_executable.model, "end_layer"):
        end_layer = model_executable.model.end_layer
    else:
        end_layer = len(kv_caches)

    # query_lens contains new KV caches that are added to vLLM.
    # so we will send them to decode instance
    # FIXME(Kuntai): This assume that all requests are prefill, which may not
    #                work for chunked prefill
    for idx, slen in enumerate(seq_lens):
        start_pos = sum(seq_lens[:idx])
        end_pos = start_pos + slen
        current_tokens = input_tokens_tensor[start_pos:end_pos]

        keys, values = [], []
        kv_tuple_list = []

        for layer_id in range(start_layer, end_layer):
            kv_cache = kv_caches[layer_id - start_layer]

            _, _, num_heads, head_size = kv_cache[0].shape

            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

            current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

            kv_tuple_list.append(
                    (key_cache[current_slot_mapping],
                    value_cache[current_slot_mapping])
                )

    
        engine.store(current_tokens, tuple(kv_tuple_list))


def lmcache_retrieve_kv(
    model_executable,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor]
) -> "ModelInputForGPUWithSamplingMetadata":
    """Retrieve the KV caches from LMCache for the current model_input. And 
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :return: The rebuilt model_input to reflect the changes in KV.
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    # This is disagg decode instance, during prefill state
    # Need to receive KV from the prefill instance
    input_tokens_tensor = model_input.input_tokens
    seq_lens = model_input.attn_metadata.seq_lens
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    input_tokens_list = []
    num_computed_tokens_list = []
    start_pos_list = []

    if hasattr(model_executable.model, "start_layer"):
        start_layer = model_executable.model.start_layer
    else:
        start_layer = 0

    if hasattr(model_executable.model, "end_layer"):
        end_layer = model_executable.model.end_layer
    else:
        end_layer = len(kv_caches)


    # enumerate different requests
    # FIXME(Kuntai): This impl assumes that all requests are prefill.
    num_request_not_found = 0
    for idx, slen in enumerate(seq_lens):

        start_pos = sum(seq_lens[:idx])
        end_pos = start_pos + slen
        current_tokens = input_tokens_tensor[start_pos:end_pos]
        num_tokens = slen

        input_tokens_list.append(current_tokens)
        start_pos_list.append(start_pos)

        kv_tuple, num_computed_tokens = engine.retrive(current_tokens)

        # Avoid the error when prefix is exactly the same as the retrieved
        if num_computed_tokens == num_tokens:
            num_computed_tokens -= 1

        if num_computed_tokens == 0:
            num_request_not_found += 1
            continue

        num_computed_tokens_list.append(num_computed_tokens)
        #is_complete = (num_computed_tokens == num_tokens)
        end_pos = start_pos + num_computed_tokens

        # receive KV cache from disaggregated prefill instance
        # TODO: this depends on model_executable has the following attributes
        # - model.layers -- list of Layer
        # - model.layers[i].self_attn
        for i in range(start_layer, end_layer):
            idx = i - start_layer

            kv_cache = kv_caches[idx]
            layer = model_executable.model.layers[i]

            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                kv_tuple[idx][0].to(key_cache.device),
                kv_tuple[idx][1].to(value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )

    if num_request_not_found == 0: # All the request can be skipped for a bit
        rebuilt_model_input = build_partial_prefill_input(
            model_input,
            input_tokens_list,
            num_computed_tokens_list,
            start_pos_list,
            slot_mapping,
            device=kv_cache[0].device,
        )
        logger.debug("Rebuilt the input!")
        return rebuilt_model_input

    logger.debug("Returning the original input!")
    return model_input

def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    input_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
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

    # recounting query and context lengths
    for idx in range(len(input_tokens_list)):
        token_tensor = input_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        start_pos = start_pos_list[idx]

        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        # TODO(Jiayi): please check the correctness of next line
        rebuilt_input_positions.append(
            model_input.input_positions[start_pos +
                                        num_computed_token : start_pos +
                                        num_token])
        q_len = num_token - num_computed_token
        rebuilt_query_lens.append(q_len)

        # Attn metadata-related
        rebuilt_num_prefills += 1
        rebuilt_num_prefill_tokens += q_len
        new_slot_mapping = slot_mapping_flat[start_pos + num_computed_token : start_pos + num_token]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
        # TODO(Jiayi): remove hard-code (block_size=16)
        blk_size = 16
        temp_block_table = [
            slot_mapping_flat[i] // blk_size
            for i in range(start_pos, start_pos + num_token, blk_size)
        ]
        rebuilt_block_tables.append(temp_block_table)
        rebuilt_query_start_loc.append(rebuilt_num_prefill_tokens)  #start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # Sampling metadata related
        #seq_groups (use rebuilt query lens)
        rebuilt_selected_token_indices.append(rebuilt_num_prefill_tokens - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(
        rebuilt_slot_mapping).to(device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = torch.tensor(
        rebuilt_block_tables,
        dtype=model_input.attn_metadata.block_tables.dtype).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None

    # rebuilt sampling_metadata
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
    )

    return rebuilt_model_input
