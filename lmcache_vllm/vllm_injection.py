"""
This version works with vllm-0.6.1.post2 and 0.6.2
"""
from functools import wraps
import torch
import os
import asyncio
import dataclasses
from typing import Optional, List

from vllm.multimodal import MultiModalInputs
from vllm.lora.request import LoRARequest
from vllm.worker.model_runner_base import dump_input_when_exception
from vllm.distributed import get_pp_group
from vllm import _custom_ops as ops

from lmcache_vllm.vllm_adapter import (lmcache_get_config,
        init_lmcache_engine, lmcache_should_store, lmcache_should_retrieve,
        lmcache_store_kv, lmcache_retrieve_kv, close_lmcache_engine,
        create_model_input_subset,
        broadcast_seq_group_metadata, StoreStatus, RetrieveStatus,
        SUPPORTED_MODELS)

from lmcache_vllm.scheduler_adapter import (_new_schedule_running, new_schedule,
                                            _new_schedule_default,
                                            new_scheduler__init__)
from lmcache_vllm.worker_base_adapter import new_worker_base_execute_model
from lmcache_vllm.llm_engine_adapter import new_step
from lmcache_vllm.gpu_executor_adapter import new_gpu_executor_execute_model
from lmcache_vllm.sequence_adapter import (NewSequenceData)

from lmcache_vllm.models.llama import inject_llama
from lmcache_vllm.attention.flash_attn import inject_flash_attn
from lmcache_vllm.attention.flash_attn_compact import inject_flash_attn_compact
from lmcache_vllm.attention.xformers_compact import inject_xformers_compact

from lmcache.compactor import (CompactorInput, CompactorOutput, CompactorMetadata,
                                LMCacheCompactorBuilder)

from lmcache.logging import init_logger
logger = init_logger(__name__)

@torch.inference_mode()
@dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
def new_execute_model(
    self,
    model_input,
    kv_caches,
    intermediate_tensors,
    compactor_input = None,
    num_steps: int = 1,
): 
    # TODO: Currently, lmcache is disabled in memory compaction 
    # Please make memory compaction consistent with lmcache
    if os.getenv("LMC_COMPACTOR", None) is None:
        init_lmcache_engine(self.model_config, self.parallel_config, self.cache_config)
    

        # TODO(Jiayi): broadcast the necessary `seq_group_metadata` in every model
        # execution. Maybe there's a more efficient way.
        model_input = broadcast_seq_group_metadata(model_input, self.is_driver_worker)
    
    model_input_subset = create_model_input_subset(
        self.model_config.model, self.model)
    
    
    # TODO(Jiayi): squash the following function calls into 
    # something like "pre_model_update"
    
    # TODO(Jiayi): clean up memory according to end_seq_ids
    # Preallocate memory for imp_scores
    if os.getenv("LMC_COMPACTOR", None) == "True":
        is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
        compactor_type = os.getenv("COMPACTOR_TYPE", None)
        rotary_emb = self.model.model.layers[0].self_attn.rotary_emb
        compactor_metadata = CompactorMetadata(
            num_gpu_blocks=None,
            rotary_emb=rotary_emb,
        )
        lmcache_compactor = LMCacheCompactorBuilder.get_or_create(
                                instance_id="lmcache_compactor",
                                compactor_type=compactor_type,
                                compactor_metadata=compactor_metadata)
        
        if not is_profile_run:
            lmcache_compactor.allocate_imp_scores(model_input)
    
        if compactor_input is not None:
            # LMCache memory compaction (move kv cache)
            lmcache_compactor.compact_memory(
                model_input_subset,
                kv_caches,
                compactor_input.dst_slot_mappings,
                compactor_input.preempt_seq_id)
            compactor_input.preempt_seq_id = [] 

            # LMCache memory compaction (move kv cache)
            lmcache_compactor.clean_request_states(
                compactor_input.end_seq_ids,
            )
    

    # LMCache retrieval
    is_skip = False
    if os.getenv("LMC_COMPACTOR", None) is None:
        retrieve_status = lmcache_should_retrieve(model_input, kv_caches)
        if retrieve_status != RetrieveStatus.NONE:
            logger.info(f"KV cache retrieving mode: {retrieve_status}")
            model_input, is_skip = lmcache_retrieve_kv(
                self.model, model_input, 
                model_input_subset, kv_caches, retrieve_status)
            if is_skip:
                logger.debug("Prefill is entirely skipped")
                
                # Create a dummy hiddens_states
                num_tok = len(model_input.input_tokens)
                num_dim = self.model.model.embed_tokens.embedding_dim
                hidden_or_intermediate_states = torch.ones(
                    num_tok, num_dim,
                    device=model_input.input_tokens.device,
                    dtype=self.model.model.embed_tokens.weight.dtype)
                
    
    # TODO(Jiayi): Currently, we do not handle the last chunk in chunk prefill
    
    if num_steps > 1:
        raise ValueError("num_steps > 1 is not supported in ModelRunner")
 
    if self.lora_config:
        assert model_input.lora_requests is not None
        assert model_input.lora_mapping is not None
        self.set_active_loras(model_input.lora_requests,
                              model_input.lora_mapping)
 
    if self.prompt_adapter_config:
        assert model_input.prompt_adapter_requests is not None
        assert model_input.prompt_adapter_mapping is not None
        self.set_active_prompt_adapters(
            model_input.prompt_adapter_requests,
            model_input.prompt_adapter_mapping)
 
    self.attn_state.begin_forward(model_input)
 
    # Currently cuda graph is only supported by the decode phase.
    assert model_input.attn_metadata is not None
    prefill_meta = model_input.attn_metadata.prefill_metadata
    decode_meta = model_input.attn_metadata.decode_metadata
    # TODO(andoorve): We can remove this once all
    # virtual engines share the same kv cache.
    virtual_engine = model_input.virtual_engine
    if prefill_meta is None and decode_meta.use_cuda_graph:
        assert model_input.input_tokens is not None
        graph_batch_size = model_input.input_tokens.shape[0]
        model_executable = self.graph_runners[virtual_engine][
            graph_batch_size]
    else:
        model_executable = self.model

    multi_modal_kwargs = model_input.multi_modal_kwargs or {}
    seqlen_agnostic_kwargs = {
        "finished_requests_ids": model_input.finished_requests_ids,
        "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
    } if self.has_seqlen_agnostic else {}
    if (self.observability_config is not None
            and self.observability_config.collect_model_forward_time):
        model_forward_start = torch.cuda.Event(enable_timing=True)
        model_forward_end = torch.cuda.Event(enable_timing=True)
        model_forward_start.record()

    if not is_skip:
        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                        device=self.device),
            **seqlen_agnostic_kwargs)

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # LMCache storing
        if os.getenv("LMC_COMPACTOR", None) is None:
            store_status = lmcache_should_store(model_input, kv_caches)
            if any([status != StoreStatus.NONE for status in store_status]):
                logger.info(f"KV cache saving mode: {store_status}")
                lmcache_store_kv(model_executable, model_input, self.cache_config,
                                kv_caches, store_status)

    # CacheBlend updates
    if os.getenv("LMC_COMPACTOR", None) is None and \
            lmcache_get_config().enable_blending and \
            hasattr(model_input.attn_metadata, "blend_metadata") and \
            model_input.attn_metadata.blend_metadata.selected_token_indices is not None:
        new_selected_token_indices = \
                model_input.attn_metadata.blend_metadata.selected_token_indices
        model_input.sampling_metadata.selected_token_indices = \
                new_selected_token_indices
        logger.debug(f"Updating selected_token_indices to {new_selected_token_indices} after blending")

    # Compute the logits in the last pipeline stage.
    if not get_pp_group().is_last_rank:
        if (self.is_driver_worker
                and hidden_or_intermediate_states is not None
                and isinstance(hidden_or_intermediate_states,
                               IntermediateTensors)
                and self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.synchronize()
            model_forward_time = model_forward_start.elapsed_time(
                model_forward_end)
            orig_model_forward_time = 0.0
            if intermediate_tensors is not None:
                orig_model_forward_time = intermediate_tensors.tensors.get(
                    "model_forward_time", torch.tensor(0.0)).item()
            hidden_or_intermediate_states.tensors["model_forward_time"] = (
                torch.tensor(model_forward_time + orig_model_forward_time))
        return hidden_or_intermediate_states
 
    logits = self.model.compute_logits(hidden_or_intermediate_states,
                                       model_input.sampling_metadata)

    
    if not self.is_driver_worker:
        return []

    # Jiayi: this call back calls `_process_model_outputs`
    # in vllm/engine/llm_engine.py
    if model_input.async_callback is not None:
        model_input.async_callback()
 
    # Sample the next token.
    output: SamplerOutput = self.model.sample(
        logits=logits,
        sampling_metadata=model_input.sampling_metadata,
    )
    
    if (self.observability_config is not None
            and self.observability_config.collect_model_forward_time
            and output is not None):
        model_forward_end.synchronize()
        model_forward_time = model_forward_start.elapsed_time(
            model_forward_end)
        orig_model_forward_time = 0.0
        if intermediate_tensors is not None:
            orig_model_forward_time = intermediate_tensors.tensors.get(
                "model_forward_time", torch.tensor(0.0)).item()
        # If there are multiple workers, we are still tracking the latency
        # from the start time of the driver worker to the end time of the
        # driver worker. The model forward time will then end up covering
        # the communication time as well.
        output.model_forward_time = (orig_model_forward_time +
                                     model_forward_time)
 
    if self.return_hidden_states:
        # we only need to pass hidden states of most recent token
        assert model_input.sampling_metadata is not None
        indices = model_input.sampling_metadata.selected_token_indices
        if model_input.is_prompt:
            hidden_states = hidden_or_intermediate_states.index_select(
                0, indices)
            output.prefill_hidden_states = hidden_or_intermediate_states
        elif decode_meta.use_cuda_graph:
            hidden_states = hidden_or_intermediate_states[:len(indices)]
        else:
            hidden_states = hidden_or_intermediate_states
 
        output.hidden_states = hidden_states

    
    # lmcache_compactor post model actions
    compactor_output = None
    if os.getenv("LMC_COMPACTOR", None) == "True":
        compactor_output = lmcache_compactor.post_model_update(
                                kv_caches, model_input)
    
    return [output], compactor_output

def _patch_padding_space(
    tokenizer_id: str,
    prompt: str,
) -> str:
    """
    patch padding tokens to enable caching decode KV cache
    """
    if tokenizer_id in SUPPORTED_MODELS.mistral_family:
        prompt = prompt.replace("[/INST]  ", "[/INST] ")
    elif tokenizer_id in SUPPORTED_MODELS.llama_family:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif tokenizer_id in SUPPORTED_MODELS.glm_family:
        prompt += "<|assistant|>\n"
    return prompt

def _new_tokenize_prompt(
    self,
    prompt: str,
    request_id: str,
    lora_request: Optional[LoRARequest],
) -> List[int]:
    """
    Apply the model's tokenizer to a text prompt, returning the
    corresponding token IDs.
    """
    tokenizer = self.get_tokenizer_group()

    # Jiayi: Patch starts here
    tokenizer_id = tokenizer.tokenizer_id
    prompt = _patch_padding_space(tokenizer_id, prompt)
    # Jiayi: Patch ends here

    res = tokenizer.encode(request_id=request_id,
                            prompt=prompt,
                            lora_request=lora_request)
    
    return res

async def _new_tokenize_prompt_async(
    self,
    prompt: str,
    request_id: str,
    lora_request: Optional[LoRARequest],
) -> List[int]:
    """Async version of :meth:`_tokenize_prompt`."""
    
    tokenizer = self.get_tokenizer_group()
    
    # Jiayi: Patch starts here
    tokenizer_id = tokenizer.tokenizer_id
    prompt = _patch_padding_space(tokenizer_id, prompt)
    # Jiayi: Patch ends here

    res = await tokenizer.encode_async(request_id=request_id,
                                        prompt=prompt,
                                        lora_request=lora_request)
    
    return res

def new_log_task_completion(task: asyncio.Task,
                            error_callback) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        raise AssertionError(
            f"The engine background task should never finish without an "
            f"exception. {return_value}")
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        close_lmcache_engine()
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise RuntimeError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on Github. See stack trace above for the "
            "actual cause.") from e

original_prepare_model_input = None
def wrap_prepare_model_input(
        self,
        seq_group_metadata_list,
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ):
    """Wrap prepare_model_input to put seq_group_metadata_list
    into model_input.
    """
    global original_prepare_model_input
    model_input = original_prepare_model_input(
        self, seq_group_metadata_list, virtual_engine, finished_requests_ids)

    # NOTE(Sixian): Use seq_group_metadata_list because
    # sampling_metadata is only available
    # at the last stage of pipeline parallelism stages.
    return dataclasses.replace(model_input, seq_group_metadata_list=seq_group_metadata_list)

def InitLMCacheEnvironment() -> None:
    """Initialize the LMCache environment.
    """
    
    import vllm.worker.model_runner 
    vllm.worker.model_runner.ModelRunner.execute_model = new_execute_model

    import vllm.engine.async_llm_engine
    vllm.engine.async_llm_engine._log_task_completion = new_log_task_completion
    
    import vllm.worker.model_runner
    global original_prepare_model_input
    original_prepare_model_input = vllm.worker.model_runner.ModelRunner.prepare_model_input
    vllm.worker.model_runner.ModelRunner.prepare_model_input = wrap_prepare_model_input
    
    import vllm
    vllm.inputs.preprocess.InputPreprocessor._tokenize_prompt = _new_tokenize_prompt
    vllm.inputs.preprocess.InputPreprocessor._tokenize_prompt_async = _new_tokenize_prompt_async
    
    # inject scheduler
    import vllm.core.scheduler
    vllm.core.scheduler.Scheduler._schedule_running = _new_schedule_running
    vllm.core.scheduler.Scheduler._schedule_default = _new_schedule_default
    vllm.core.scheduler.Scheduler.schedule = new_schedule
    vllm.core.scheduler.Scheduler.__init__ = new_scheduler__init__
    
    # inject llm_engine
    import vllm.engine.llm_engine
    vllm.engine.llm_engine.LLMEngine.step = new_step
    
    # inject gpu_executor
    import vllm.executor.gpu_executor
    vllm.executor.gpu_executor.GPUExecutor.execute_model = new_gpu_executor_execute_model
    
    # inject worker_base
    import vllm.worker.worker_base
    vllm.worker.worker_base.LocalOrDistributedWorkerBase.execute_model = new_worker_base_execute_model
    
    # inject vllm sequence
    import vllm.sequence
    #vllm.sequence.Sequence.__init__ = new_sequence__init__
    #vllm.sequence.SequenceData.append_token_id = new_sequencedata_append_token_id
    #vllm.sequence.SequenceData.get_len = new_sequencedata_get_len
    vllm.sequence.SequenceData = NewSequenceData
    
    # inject flash attention
    # inject_flash_attn_compact()
    
    # inject xformers attention
    inject_xformers_compact()
    
    # Cacheblend
    if lmcache_get_config().enable_blending:
        inject_llama()
        inject_flash_attn()
