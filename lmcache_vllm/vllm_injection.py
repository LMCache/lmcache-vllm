"""
This version works with vllm-0.6.1.post2, 0.6.2 and 0.7.3+
"""
import torch
import asyncio
import dataclasses
from dataclasses import fields
from typing import Optional, List, Set, Dict, Any, Union, AsyncGenerator
import inspect

from vllm.multimodal import MultiModalKwargs as MultiModalInputs
from vllm.lora.request import LoRARequest

from vllm.distributed import get_pp_group
from vllm.transformers_utils.tokenizer import MistralTokenizer

from lmcache_vllm.vllm_adapter import (init_lmcache_engine, 
        lmcache_should_store, lmcache_should_retrieve,
        lmcache_store_kv, lmcache_retrieve_kv, close_lmcache_engine,
        broadcast_seq_group_metadata,
        lmcache_remove_request_id_indices, StoreStatus, RetrieveStatus,
        SUPPORTED_MODELS)
from lmcache_vllm.lmcache_utils import lmcache_get_config
from lmcache_vllm.blend_adapter import attach_blend_prompt_indices, get_blend_separator, add_blend_indices

from lmcache_vllm.models.llama import inject_llama
from lmcache_vllm.attention.flash_attn import inject_flash_attn
from vllm.transformers_utils.tokenizer import AnyTokenizer
# Handle different TokensPrompt locations in different vLLM versions
try:
    from vllm.inputs import TokensPrompt
except ImportError:
    try:
        from vllm.inputs.preprocess import TokensPrompt
    except ImportError:
        from vllm.entrypoints.openai.serving_engine import TextTokensPrompt as TokensPrompt

from vllm.entrypoints.openai.serving_engine import AnyRequest
from pydantic import Field
from typing_extensions import Annotated

from lmcache.logging import init_logger
logger = init_logger(__name__)

from vllm.forward_context import set_forward_context

@torch.inference_mode()
def new_execute_model(
    self,
    model_input,
    kv_caches,
    intermediate_tensors=None,
    num_steps=1,
    **kwargs
): 
    # Retrieve the cached list
    seq_group_metadata_list = getattr(self, '_last_seq_group_metadata_list', [])

    # Skip LMCache operations during profiling
    if kv_caches is None:
        logger.debug("Profiling run detected, skipping LMCache operations")
        if self.lora_config:
            self.set_active_loras(model_input.lora_requests, model_input.lora_mapping)
        if self.prompt_adapter_config:
            self.set_active_prompt_adapters(model_input.prompt_adapter_requests, model_input.prompt_adapter_mapping)
        self.attn_state.begin_forward(model_input)
        return self.model(input_ids=model_input.input_tokens, positions=model_input.input_positions,
                         kv_caches=None, attn_metadata=model_input.attn_metadata,
                         intermediate_tensors=intermediate_tensors,
                         **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs or {}, device=self.device))

    if not seq_group_metadata_list:
        logger.warning("seq_group_metadata_list not found in cache, proceeding cautiously.")

    init_lmcache_engine(self.model_config, self.parallel_config, self.cache_config)

    # Pass the retrieved list explicitly
    model_input = broadcast_seq_group_metadata(model_input, self.is_driver_worker, seq_group_metadata_list)
    
    # LMCache retrieval
    # Pass the list explicitly to functions needing it
    retrieve_status = lmcache_should_retrieve(model_input, kv_caches, seq_group_metadata_list)
    is_skip = False
    if retrieve_status != RetrieveStatus.NONE:
        logger.info(f"KV cache retrieving mode: {retrieve_status}")
        model_input, is_skip = lmcache_retrieve_kv(
            self.model, self.model_config.model, model_input, kv_caches, retrieve_status, seq_group_metadata_list=seq_group_metadata_list)
        if is_skip:
            logger.debug("Prefill is entirely skipped")
            
            # Handle vLLM 0.8.4's graph batch size adjustments for previous_hidden_states
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                    dtype=previous_hidden_states.dtype,
                    device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model

        # Handle vLLM 0.8.4's KV cache transfer functionality
        bypass_model_exec = False
        if hasattr(self, "need_recv_kv") and self.need_recv_kv(model_input, kv_caches):
            from vllm.distributed import get_kv_transfer_group
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        # Handle all the kwargs for model execution
        multi_modal_kwargs = model_input.multi_modal_kwargs or {}

        # Check which seqlen_agnostic attribute is used
        if hasattr(self, 'has_inner_state'):
            seqlen_agnostic_condition = self.has_inner_state
        else:
            seqlen_agnostic_condition = getattr(self, 'has_seqlen_agnostic', False)
            
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
    if (self.observability_config is not None
            and self.observability_config.collect_model_forward_time):
        model_forward_start = torch.cuda.Event(enable_timing=True)
        model_forward_end = torch.cuda.Event(enable_timing=True)
        model_forward_start.record()

    # Determine if profiling (heuristic: kv_caches is None)
    is_profiling = kv_caches is None

    # Create forward context
    # Ensure attn_metadata exists, even if basic, for context creation
    if model_input.attn_metadata is None:
        # This case might happen in edge scenarios or if prepare_model_input changed.
        # We might need a default/dummy AttentionMetadata here if required by ForwardContext.
        # For now, let's assert it should exist after prepare_model_input.
        assert False, "AttentionMetadata is None in execute_model, cannot create ForwardContext."
        

    if not is_skip:
        # Execute model within the forward context
        with set_forward_context(
            attn_metadata=model_input.attn_metadata,
            vllm_config=self.vllm_config,
            virtual_engine=model_input.virtual_engine,
            ):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                            device=self.device)
            )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # LMCache storing
        # Pass the list explicitly to functions needing it
        store_status = lmcache_should_store(model_input, kv_caches, seq_group_metadata_list)
        if any([status != StoreStatus.NONE for status in store_status]):
            lmcache_store_kv(self.model_config, self.parallel_config, model_executable,
                    model_input, self.cache_config, kv_caches, store_status, seq_group_metadata_list)

    # CacheBlend updates
    if lmcache_get_config().enable_blending and \
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

        return [output]
    except Exception as e:
        # Add manual error handling if dump_input_when_exception is not available
        if not has_dump_decorator:
            logger.error(f"Error in execute_model: {str(e)}", exc_info=True)
            logger.error(f"Model input: {model_input}")
        raise

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
    # Sixian: Patch starts here.
    cache_config = lmcache_get_config()
    if cache_config.enable_blending:
        input_ids = []
        blend_indices = []
        is_first_chunk = True
        current_idx = 0
        blend_separator = get_blend_separator()
        text_chunk_list = prompt.split(blend_separator)
        for text_chunk in text_chunk_list:
            force_no_special_tokens = not is_first_chunk and not cache_config.blend_add_special_in_precomp
            effective_add_special_tokens = not force_no_special_tokens
            encoded = tokenizer.encode(request_id=request_id,
                                       prompt=text_chunk,
                                       lora_request=lora_request,
                                       add_special_tokens=effective_add_special_tokens)
            input_ids.extend(encoded)
            current_idx += len(encoded)
            blend_indices.append(current_idx)
            is_first_chunk = False
        if len(blend_indices) > 0:
            blend_indices.pop()
        add_blend_indices(request_id, blend_indices)
        return input_ids
    # Sixian: Patch ends here.
    else:
        return tokenizer.encode(request_id=request_id,
                                prompt=prompt,
                                lora_request=lora_request)
    

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

    # Sixian: Patch starts here.
    cache_config = lmcache_get_config()
    if cache_config.enable_blending:
        input_ids = []
        blend_indices = []
        is_first_chunk = True
        current_idx = 0
        blend_separator = get_blend_separator()
        text_chunk_list = prompt.split(blend_separator)
        for text_chunk in text_chunk_list:
            force_no_special_tokens = not is_first_chunk and not cache_config.blend_add_special_in_precomp
            effective_add_special_tokens = not force_no_special_tokens
            encoded = await tokenizer.encode_async(request_id=request_id,
                                                   prompt=text_chunk,
                                                   lora_request=lora_request,
                                                   add_special_tokens=effective_add_special_tokens)
            input_ids.extend(encoded)
            current_idx += len(encoded)
            blend_indices.append(current_idx)
            is_first_chunk = False
        if len(blend_indices) > 0:
            blend_indices.pop()
        add_blend_indices(request_id, blend_indices)
        return input_ids
    # Sixian: Patch ends here.
    else:
        return await tokenizer.encode_async(request_id=request_id,
                                            prompt=prompt,
                                            lora_request=lora_request)

def _new_normalize_prompt_text_to_input(
    self,
    request: AnyRequest,
    tokenizer: AnyTokenizer,
    prompt: str,
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    add_special_tokens: bool,
) -> TokensPrompt:

    # Jiayi: Patch starts here
    
    # (HACK) Check if tokenizer has a name_or_path attribute
    if hasattr(tokenizer, "name_or_path"):
        tokenizer_id = tokenizer.name_or_path
    elif hasattr(tokenizer, "mistral"):
        tokenizer_id = "mistral"
    else:
        logger.warning("Tokenizer does not have a name_or_path attribute")
        logger.info("Using default tokenizer_id (LLama)")
        tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct"
        
    prompt = _patch_padding_space(tokenizer_id, prompt)
    # Jiayi: Patch ends here

    # Sixian: Patch starts here.
    cache_config = lmcache_get_config()
    if cache_config.enable_blending:
        blend_separator = get_blend_separator()
        text_chunk_list = prompt.split(blend_separator)
        input_ids = []
        input_text = ""
        blend_indices = []
        current_idx = 0
        is_first_chunk = True
        for text_chunk in text_chunk_list:
            force_no_special_tokens = not is_first_chunk and not cache_config.blend_add_special_in_precomp
            effective_add_special_tokens = add_special_tokens and not force_no_special_tokens
            if truncate_prompt_tokens is None:
                encoded = tokenizer(text_chunk, add_special_tokens=effective_add_special_tokens)
            else:
                encoded = tokenizer(text_chunk, 
                                    add_special_tokens=effective_add_special_tokens,
                                    truncation=True,
                                    max_length=truncate_prompt_tokens)
            input_ids.extend(encoded.input_ids)
            input_text += text_chunk
            current_idx += len(encoded.input_ids)
            blend_indices.append(current_idx)
            is_first_chunk = False
        text_tokens_prompt: TokensPrompt = self._validate_input(request, input_ids, input_text)
        if len(blend_indices) > 0:
            blend_indices.pop()
        text_tokens_prompt["blend_indices"] = blend_indices
        return text_tokens_prompt
    # Sixian: Patch ends here.
    else:
        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        else:
            encoded = tokenizer(prompt,
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=truncate_prompt_tokens)
        
        input_ids = encoded.input_ids
        
        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

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

    # Store the list for later use in new_execute_model
    self._last_seq_group_metadata_list = seq_group_metadata_list

    # NOTE(Sixian): Use seq_group_metadata_list because
    # sampling_metadata is only available
    # at the last stage of pipeline parallelism stages.
    return model_input

original_prepare_model_input_tensors = None
def wrap_prepare_model_input_tensors(
        self,
        seq_group_metadata_list,
        finished_requests_ids: Optional[List[str]] = None
    ):
    model_input = original_prepare_model_input_tensors(self,
        seq_group_metadata_list, finished_requests_ids)
    attn_metadata = model_input.attn_metadata
    if attn_metadata is not None:
        if lmcache_get_config().enable_blending:
            attach_blend_prompt_indices(seq_group_metadata_list, attn_metadata)
    return model_input

def new_free_finished_seqs(self, seq_group) -> None:
    """Free finished seqs in a sequence group."""
    for seq in seq_group.get_seqs():
        if seq.is_finished():
            self.free_seq(seq)
    if seq_group.is_finished():
        lmcache_remove_request_id_indices(seq_group.request_id)


def new_asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        result = {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }
        if hasattr(self, "blend_metadata"):
            result["blend_metadata"] = self.blend_metadata
        return result


@classmethod
def new_from_broadcasted_tensor_dict_with_sampling(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        from vllm.worker.model_runner_base import _init_sampling_metadata_from_tensor_dict, _init_attn_metadata_from_tensor_dict
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        if "blend_metadata" in tensor_dict:
            assert "attn_metadata" in tensor_dict
            setattr(tensor_dict["attn_metadata"], "blend_metadata", tensor_dict["blend_metadata"])
            tensor_dict.pop("blend_metadata")

        return cls(**tensor_dict)


original_extract_prompt_components = None
original_extract_prompt_components_async = None

def new_extract_prompt_components(self,
                                  inputs,
                                  request_id,
                                  lora_request = None):
    # If passed in the form of token_ids, should have blend_indices.
    # else should handle in tokenize_prompt.
    if isinstance(inputs, dict):
        if "blend_indices" in inputs:
            blend_indices = inputs.pop("blend_indices")
            add_blend_indices(request_id, blend_indices)
    prompt, prompt_token_ids, multi_modal_data = original_extract_prompt_components(self, inputs, request_id, lora_request)
    return prompt, prompt_token_ids, multi_modal_data

async def new_extract_prompt_components_async(
        self,
        inputs,
        request_id,
        lora_request = None,
    ):
    if isinstance(inputs, dict):
        if "blend_indices" in inputs:
            blend_indices = inputs.pop("blend_indices")
            add_blend_indices(request_id, blend_indices)
    prompt, prompt_token_ids, multi_modal_data = await original_extract_prompt_components_async(
        self, inputs, request_id, lora_request)
    return prompt, prompt_token_ids, multi_modal_data

original_llm_engine_init = None
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.usage.usage_lib import UsageContext
def new_llm_engine_init(
        self,
        vllm_config: VllmConfig,
        executor_class,
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
    if use_cached_outputs:
        original_llm_engine_init(self,
                                vllm_config,
                                executor_class,
                                log_stats,
                                usage_context,
                                stat_loggers,
                                input_registry,
                                use_cached_outputs)
    else:
        original_llm_engine_init(self,
                                vllm_config,
                                executor_class,
                                log_stats,
                                usage_context,
                                stat_loggers,
                                input_registry,
                                use_cached_outputs)

    init_lmcache_engine(vllm_config.model_config, vllm_config.parallel_config, vllm_config.cache_config)


def new_tokenizer_group_encode(self,
                         prompt: str,
                         request_id: Optional[str] = None,
                         lora_request: Optional[LoRARequest] = None,
                         add_special_tokens: bool = True) -> List[int]:
    tokenizer = self.get_lora_tokenizer(lora_request)
    # If the function accepts add_special_tokens, we should pass it
    # It defaults to True.
    signature = inspect.signature(tokenizer.encode)
    if "add_special_tokens" in signature.parameters:
        ret = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    else:
        assert type(tokenizer) == MistralTokenizer
        # NOTE: MistralTokenizer always sets second value to True in encode
        # even if add_special_tokens is False, here I changed the behavior.
        ret = tokenizer.tokenizer.encode(prompt, bos=add_special_tokens, eos=False)
    self._raise_if_input_too_long(ret, lora_request)
    return ret


async def new_tokenizer_group_encode_async(self,
                                    prompt: str,
                                    request_id: Optional[str] = None,
                                    lora_request: Optional[LoRARequest] = None,
                                    add_special_tokens: bool = True) -> List[int]:
    tokenizer = await self.get_lora_tokenizer_async(lora_request)
    signature = inspect.signature(tokenizer.encode)
    if "add_special_tokens" in signature.parameters:
        ret = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    else:
        assert type(tokenizer) == MistralTokenizer
        ret = tokenizer.tokenizer.encode(prompt, bos=add_special_tokens, eos=False)
    self._raise_if_input_too_long(ret, lora_request)
    return ret

# Add custom implementation for iterate_with_cancellation as it's no longer in vLLM 0.7.3
async def iterate_with_cancellation(iterator, is_cancelled):
    """Iterate through an async iterator until a cancellation event is set.
    When the cancellation event is triggered, we will try our best to clean up.
    """
    try:
        async for item in iterator:
            if await is_cancelled():
                # Early terminate the request
                break
            yield item
    except Exception as e:
        # Re-raises the exception
        raise e

from vllm.entrypoints.openai.serving_chat import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, RequestResponseMetadata, Request
from vllm.entrypoints.chat_utils import parse_chat_messages_futures, apply_mistral_chat_template, apply_hf_chat_template
from vllm.utils import random_uuid
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.inputs import TokensPrompt
async def new_create_chat_completion(
    self,
    request: ChatCompletionRequest,
    raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
        ErrorResponse]:
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI
    ChatCompletion API.

    """
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        logger.error("Error with model %s", error_check_ret)
        return error_check_ret

    # If the engine is dead, raise the engine's DEAD_error.
    # This is required for the streaming case, where we return a
    # success status before we actually start generating text :).
    if self.engine_client.errored:
        raise self.engine_client.dead_error

    try:
        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        model_config = self.model_config
        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        # Handle differences between vLLM 0.6.x and 0.8.4
        if has_old_chat_api:
            # Old API in vLLM 0.6.x
            conversation, mm_data_future = parse_chat_messages_futures(
                request.messages, model_config, tokenizer)

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            prompt: Union[str, List[int]]
            is_mistral_tokenizer = isinstance(tokenizer, MistralTokenizer)
            if is_mistral_tokenizer:
                prompt = apply_mistral_chat_template(
                    tokenizer,
                    messages=request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    tools=tool_dicts,
                    documents=request.documents,
                    **(request.chat_template_kwargs or {}),
                )
            else:
                prompt = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    tools=tool_dicts,
                    documents=request.documents,
                    **(request.chat_template_kwargs or {}),
                )
        else:
            # New API in vLLM 0.8.4
            try:
                # Use the _preprocess_chat method which is now part of the class
                conversation, request_prompts, engine_prompts = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    request.chat_template or self.chat_template,
                    "text",  # chat_template_content_format
                    request.add_generation_prompt,
                    tool_dicts=[tool.model_dump() for tool in request.tools] if request.tools else None,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
                # In vLLM 0.8.4, engine_prompts contains the prepared prompts
                if engine_prompts:
                    prompt = engine_prompts[0]
                else:
                    raise ValueError("No engine prompts generated")

                # Since mm_data isn't directly accessible, try to extract it from the prompt
                mm_data_future = asyncio.Future()
                if isinstance(prompt, dict) and "multi_modal_data" in prompt:
                    mm_data_future.set_result(prompt.get("multi_modal_data"))
                else:
                    mm_data_future.set_result(None)
            except (AttributeError, TypeError) as e:
                # Fall back to original method but without the parse_chat_messages_futures 
                return await super().create_chat_completion(request, raw_request)
    except Exception as e:
        logger.error("Error in applying chat template from request: %s", e)
        return self.create_error_response(str(e))

    try:
        mm_data = await mm_data_future
    except Exception as e:
        logger.error("Error in loading multi-modal data: %s", e)
        return self.create_error_response(str(e))

    # validation for OpenAI tools
    # tool_choice = "required" is not supported
    if request.tool_choice == "required":
        return self.create_error_response(
            "tool_choice = \"required\" is not supported!")

    if not isinstance(tokenizer, MistralTokenizer) and request.tool_choice == "auto" and not (
            self.enable_auto_tools and self.tool_parser is not None):
        # for hf tokenizers, "auto" tools requires
        # --enable-auto-tool-choice and --tool-call-parser
        return self.create_error_response(
            "\"auto\" tool choice requires "
            "--enable-auto-tool-choice and --tool-call-parser to be set")

    request_id = f"chat-{random_uuid()}"

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    try:
        # For vLLM 0.8.4 compatibility
        if hasattr(self, "_guided_decode_logits_processor"):
            guided_decode_logits_processor = (
                await self._guided_decode_logits_processor(request, tokenizer))
        else:
            guided_decode_logits_processor = None

        if isinstance(prompt, str):
            prompt_inputs = self._tokenize_prompt_input(
                request,
                tokenizer,
                prompt,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        elif isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            # vLLM 0.8.4 might provide prompt as a dictionary
            prompt_inputs = prompt
        else:
            assert isinstance(prompt, list) and isinstance(
                prompt[0], int
            ), "Prompt has to be either a string, dictionary with prompt_token_ids, or a list of token ids"
            prompt_inputs = {"prompt": tokenizer.decode(prompt), "prompt_token_ids": prompt}

        assert prompt_inputs is not None

        if hasattr(request, "to_sampling_params"):
            # vLLM 0.6.x API
            sampling_params = request.to_sampling_params(
                tokenizer,
                guided_decode_logits_processor,
                default_max_tokens=self.max_model_len -
                len(prompt_inputs["prompt_token_ids"]))
        else:
            # vLLM 0.8.4 API
            from vllm.sampling_params import SamplingParams

            # Create sampling params from the request attributes
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                use_beam_search=request.use_beam_search,
                stop=request.stop,
                stop_token_ids=request.stop_token_ids,
                max_tokens=request.max_tokens or self.max_model_len - len(prompt_inputs["prompt_token_ids"]),
                best_of=request.best_of,
                seed=request.seed,
                skip_special_tokens=request.skip_special_tokens,
                logprobs=request.logprobs,
                prompt_logprobs=request.prompt_logprobs,
                guided_logits_processor=guided_decode_logits_processor,
            )

        self._log_inputs(request_id,
                        prompt_inputs,
                        params=sampling_params,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request)

        # Get engine_inputs based on vLLM version
        if hasattr(self, "tokensPrompt") and hasattr(self.tokensPrompt, "__call__"):
            # vLLM 0.8.4 style
            from vllm.inputs import TokensPrompt
            engine_inputs = TokensPrompt(
                prompt_token_ids=prompt_inputs["prompt_token_ids"])
        else:
            # vLLM 0.6.x style
            from vllm.inputs.preprocess import TokensPrompt
            engine_inputs = TokensPrompt(
                prompt_token_ids=prompt_inputs["prompt_token_ids"])

        # Sixian: Patch starts here.
        if "blend_indices" in prompt_inputs:
            engine_inputs["blend_indices"] = prompt_inputs["blend_indices"]
        # Sixian: Patch ends here.
        if mm_data is not None:
            engine_inputs["multi_modal_data"] = mm_data

        is_tracing_enabled = (await
                            self.engine_client.is_tracing_enabled())
        trace_headers = None
        if is_tracing_enabled and raw_request:
            trace_headers = extract_trace_headers(raw_request.headers)
        if (not is_tracing_enabled and raw_request
                and contains_trace_headers(raw_request.headers)):
            log_tracing_disabled_warning()

        result_generator = self.engine_client.generate(
            engine_inputs,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
        )
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))

    if raw_request:
        result_generator = iterate_with_cancellation(
            result_generator, raw_request.is_disconnected)

    # Streaming response
    if request.stream:
        return self.chat_completion_stream_generator(
            request, result_generator, request_id, conversation, tokenizer,
            request_metadata)

    try:
        return await self.chat_completion_full_generator(
            request, result_generator, request_id, conversation, tokenizer,
            request_metadata)
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))
    

def inject_blend():
    import vllm.attention.backends.abstract
    vllm.attention.backends.abstract.AttentionMetadata.asdict_zerocopy = new_asdict_zerocopy

    import vllm.worker.model_runner
    vllm.worker.model_runner.ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict = \
    new_from_broadcasted_tensor_dict_with_sampling

    import vllm.inputs.preprocess
    global original_extract_prompt_components
    global original_extract_prompt_components_async
    original_extract_prompt_components = vllm.inputs.preprocess.InputPreprocessor._prompt_to_llm_inputs
    original_extract_prompt_components_async = vllm.inputs.preprocess.InputPreprocessor._prompt_to_llm_inputs_async
    vllm.inputs.preprocess.InputPreprocessor._extract_prompt_components = new_extract_prompt_components
    vllm.inputs.preprocess.InputPreprocessor._extract_prompt_components_async = new_extract_prompt_components_async
    from vllm.transformers_utils.tokenizer_group.tokenizer_group import TokenizerGroup
    TokenizerGroup.encode = new_tokenizer_group_encode
    TokenizerGroup.encode_async = new_tokenizer_group_encode_async
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    OpenAIServingChat.create_chat_completion = new_create_chat_completion



def InitLMCacheEnvironment() -> None:
    """Initialize the LMCache environment.
    """
    # Initialize LLMEngine
    import vllm.engine.llm_engine
    global original_llm_engine_init
    original_llm_engine_init = vllm.engine.llm_engine.LLMEngine.__init__
    vllm.engine.llm_engine.LLMEngine.__init__ = new_llm_engine_init
    
    # Replace execute_model
    import vllm.worker.model_runner 
    vllm.worker.model_runner.ModelRunner.execute_model = new_execute_model

    # Replace task completion handler
    import vllm.engine.async_llm_engine
    vllm.engine.async_llm_engine._log_task_completion = new_log_task_completion
    
    # Replace model input preparation
    import vllm.worker.model_runner
    global original_prepare_model_input
    original_prepare_model_input = vllm.worker.model_runner.ModelRunner.prepare_model_input
    vllm.worker.model_runner.ModelRunner.prepare_model_input = wrap_prepare_model_input

    global original_prepare_model_input_tensors
    original_prepare_model_input_tensors = vllm.worker.model_runner.ModelRunner._prepare_model_input_tensors
    vllm.worker.model_runner.ModelRunner._prepare_model_input_tensors = wrap_prepare_model_input_tensors

    # Replace scheduler methods
    try:
        # vLLM 0.8.4 path
        import vllm.core.scheduler
        vllm.core.scheduler.Scheduler._free_finished_seqs = new_free_finished_seqs
    except (ImportError, AttributeError):
        # Older vLLM path
        import vllm.scheduler
        vllm.scheduler.Scheduler._free_finished_seqs = new_free_finished_seqs
    
    # Replace tokenizer methods
    import vllm
    vllm.inputs.preprocess.InputPreprocessor._tokenize_prompt = _new_tokenize_prompt
    vllm.inputs.preprocess.InputPreprocessor._tokenize_prompt_async = _new_tokenize_prompt_async
    
    # Replace OpenAI serving methods
    vllm.entrypoints.openai.serving_engine.OpenAIServing._normalize_prompt_text_to_input = \
        _new_normalize_prompt_text_to_input
    
    # Cacheblend
    inject_llama()
    inject_flash_attn()
    inject_blend()
