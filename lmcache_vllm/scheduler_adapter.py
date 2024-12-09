from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStatus)
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)
from itertools import compress

from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.scheduler import (PreemptionMode, SchedulingBudget, SchedulerRunningOutputs, SchedulerOutputs,
                                 SchedulerPrefillOutputs, SchedulerSwappedInOutputs,
                                 seq_group_metadata_builder, scheduler_running_outputs_builder,
                                 scheduled_seq_group_builder)
from vllm.core.block.block_table import BlockTable
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.attention.backends.utils import compute_slot_mapping
from vllm.utils import Device, PyObjectCache

from lmcache.compactor import BaseSchedulerCompactor, CompactorInput, CompactorOutput
import time
from collections import deque
import os
from array import array

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

def new_scheduler__init__(
    self,
    scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    lora_config: Optional[LoRAConfig],
    pipeline_parallel_size: int = 1,
    output_proc_callback: Optional[Callable] = None,
) -> None:
    self.scheduler_config = scheduler_config
    self.cache_config = cache_config
    # Note for LoRA scheduling: the current policy is extremely
    # simple and NOT fair. It can lead to starvation of some
    # LoRAs. This should be improved in the future.
    self.lora_config = lora_config

    version = "v1"
    if self.scheduler_config.use_v2_block_manager:
        version = "v2"
    if self.scheduler_config.embedding_mode:
        version = "embedding"

    BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
        version)

    num_gpu_blocks = cache_config.num_gpu_blocks
    if num_gpu_blocks:
        num_gpu_blocks //= pipeline_parallel_size

    num_cpu_blocks = cache_config.num_cpu_blocks
    if num_cpu_blocks:
        num_cpu_blocks //= pipeline_parallel_size

    # Create the block space manager.
    self.block_manager = BlockSpaceManagerImpl(
        block_size=self.cache_config.block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        sliding_window=self.cache_config.sliding_window,
        enable_caching=self.cache_config.enable_prefix_caching)

    # Sequence groups in the WAITING state.
    # Contain new prefill or preempted requests.
    self.waiting: Deque[SequenceGroup] = deque()
    # Sequence groups in the RUNNING state.
    # Contain decode requests.
    self.running: Deque[SequenceGroup] = deque()
    # Sequence groups in the SWAPPED state.
    # Contain decode requests that are swapped out.
    self.swapped: Deque[SequenceGroup] = deque()
    # Sequence groups finished requests ids since last step iteration.
    # It lets the model know that any state associated with these requests
    # can and must be released after the current step.
    # This is used to evict the finished requests from the Mamba cache.
    self._finished_requests_ids: List[str] = list()
    # Time at previous scheduling step
    self.prev_time = 0.0
    # Did we schedule a prompt at previous step?
    self.prev_prompt = False
    # Latency of the last prompt step
    self.last_prompt_latency = 0.0
    # preemption mode, RECOMPUTE or SWAP
    self.user_specified_preemption_mode = scheduler_config.preemption_mode

    # The following field is test-only. It is used to inject artificial
    # preemption.
    self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
    self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                    if self.enable_artificial_preemption
                                    else 0)
    self.num_cumulative_preemption: int = 0

    # Used to cache python objects
    self._seq_group_metadata_cache: List[PyObjectCache] = []
    self._scheduler_running_outputs_cache: List[PyObjectCache] = []
    self._scheduled_seq_group_cache: List[PyObjectCache] = []

    # For async output processing, we need to swap cache buffers between
    # iterations. I.e. since the output processing is lagged one step,
    # we cannot reuse the cached objects immediately when the schedule()
    # is called again, but only when schedule() is called the second time.
    self.output_proc_callback = output_proc_callback
    self.use_async_output_proc = self.output_proc_callback is not None
    self.num_cache_iters = 2 if self.use_async_output_proc else 1

    self.cache_id = 0
    for i in range(self.num_cache_iters):
        self._seq_group_metadata_cache.append(
            PyObjectCache(seq_group_metadata_builder))
        self._scheduler_running_outputs_cache.append(
            PyObjectCache(scheduler_running_outputs_builder))
        self._scheduled_seq_group_cache.append(
            PyObjectCache(scheduled_seq_group_builder))

    # For async postprocessor, the extra decode run cannot be done
    # when the request reaches max_model_len. In this case, the request
    # will be stopped during schedule() call and added to this stop list
    # for processing and deallocation by the free_finished_seq_groups()
    self._async_stopped: List[SequenceGroup] = []

    # Jiayi Modification starts
    self.compactor_output = CompactorOutput(
        compacted_indices_dict={})
    
    self.compactor_input = CompactorInput(
        dst_slot_mappings={}, end_seq_ids=[])
    # Jiayi Modification ends


def _new_schedule_running(
    self,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    enable_chunking: bool = False,
) -> SchedulerRunningOutputs:
    """Schedule sequence groups that are running.

    Running queue should include decode and chunked prefill requests.

    Args:
        budget: The scheduling budget. The argument is in-place updated
            when any decodes are preempted.
        curr_loras: Currently batched lora request ids. The argument is
            in-place updated when any decodes are preempted.
        enable_chunking: If True, seq group can be chunked and only a
            chunked number of tokens are scheduled  if
            `budget.num_batched_tokens` has not enough capacity to schedule
            all tokens.

    Returns:
        SchedulerRunningOutputs.
    """
    ret: SchedulerRunningOutputs = \
        self._scheduler_running_outputs_cache[self.cache_id].get_object()
    ret.blocks_to_swap_out.clear()
    ret.blocks_to_copy.clear()
    ret.decode_seq_groups.clear()
    ret.prefill_seq_groups.clear()
    ret.preempted.clear()
    ret.swapped_out.clear()

    ret.num_lookahead_slots = self._get_num_lookahead_slots(
        is_prefill=False)

    ret.decode_seq_groups_list.clear()
    ret.prefill_seq_groups_list.clear()

    # Blocks that need to be swapped or copied before model execution.
    blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
    blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

    decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
    prefill_seq_groups: List[
        ScheduledSequenceGroup] = ret.prefill_seq_groups
    preempted: List[SequenceGroup] = ret.preempted
    swapped_out: List[SequenceGroup] = ret.swapped_out

    running_queue = self.running
    assert len(self._async_stopped) == 0
    
    # Jiayi Modification starts
    dst_slot_mappings = {}
    # Jiayi Modification ends
    
    while running_queue:
        seq_group = running_queue[0]
        num_running_tokens = self._get_num_new_tokens(
            seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

        if num_running_tokens == 0:
            # No budget => Stop
            break

        running_queue.popleft()

        # With async postprocessor, an extra decode run is done
        # to process the final tokens. The check below avoids this extra
        # decode run when the model max len is reached, in order to avoid
        # a memory overflow.
        if self.use_async_output_proc and seq_group.seqs[0].get_len(
        ) > self.scheduler_config.max_model_len:
            self._async_stopped.append(seq_group)
            continue

        # Jiayi Modification starts
        if os.getenv("LMC_COMPACTOR", None) == "True":
            compacted_indices_dict = self.compactor_output.compacted_indices_dict
            BaseSchedulerCompactor.compact_blocks(
                self.block_manager,
                compacted_indices_dict, 
                dst_slot_mappings,
                seq_group)
        # Jiayi Modification ends
        
        # NOTE(woosuk): Preemption happens only when there is no available
        # slot to keep all the sequence groups in the RUNNING state.
        while not self._can_append_slots(seq_group):
            budget.subtract_num_batched_tokens(seq_group.request_id,
                                            num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                    num_running_seqs)

            if (curr_loras is not None and seq_group.lora_int_id > 0
                    and seq_group.lora_int_id in curr_loras):
                curr_loras.remove(seq_group.lora_int_id)

            # Determine victim sequence
            cont_loop = True
            if running_queue:
                # Preempt the lowest-priority sequence group.
                victim_seq_group = running_queue.pop()
            else:
                # No other sequence group can be preempted.
                # Preempt the current sequence group.
                # Note: This is also where we stop this loop
                # (since there is nothing else to preempt)
                victim_seq_group = seq_group
                cont_loop = False

            # With async postprocessor, before preempting a sequence
            # we need to ensure it has no pending async postprocessor
            do_preempt = True
            if self.use_async_output_proc:
                assert self.output_proc_callback is not None
                self.output_proc_callback(
                    request_id=victim_seq_group.request_id)

                # It may be that the async pending "victim_seq_group"
                # becomes finished, in which case we simply free it.
                if victim_seq_group.is_finished():
                    self._free_finished_seq_group(victim_seq_group)
                    do_preempt = False

            # Do preemption
            if do_preempt:
                preempted_mode = self._preempt(victim_seq_group,
                                            blocks_to_swap_out)
                if preempted_mode == PreemptionMode.RECOMPUTE:
                    preempted.append(victim_seq_group)
                else:
                    swapped_out.append(victim_seq_group)

            if not cont_loop:
                break
        else:
            
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()

            scheduled_seq_group: ScheduledSequenceGroup = \
                self._scheduled_seq_group_cache[self.cache_id].get_object()
            scheduled_seq_group.seq_group = seq_group
            if is_prefill:
                scheduled_seq_group.token_chunk_size = num_running_tokens
                prefill_seq_groups.append(scheduled_seq_group)
                ret.prefill_seq_groups_list.append(seq_group)
            else:
                scheduled_seq_group.token_chunk_size = 1
                decode_seq_groups.append(scheduled_seq_group)
                ret.decode_seq_groups_list.append(seq_group)

            budget.add_num_batched_tokens(seq_group.request_id,
                                        num_running_tokens)
            # OPTIMIZATION:  Note that get_max_num_running_seqs is
            # expensive. For the default scheduling chase where
            # enable_chunking is False, num_seqs are updated before running
            # this method, so we don't have to update it again here.
            if enable_chunking:
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.add_num_seqs(seq_group.request_id, num_running_seqs)
            if curr_loras is not None and seq_group.lora_int_id > 0:
                curr_loras.add(seq_group.lora_int_id)

    self._scheduler_running_outputs_cache[self.next_cache_id].reset()
    self._scheduled_seq_group_cache[self.next_cache_id].reset()

    # Jiayi Modification starts
    self.compactor_input.dst_slot_mappings = dst_slot_mappings
    # Jiayi Modification ends
    return ret

def _new_schedule_default(self) -> SchedulerOutputs:
    """Schedule queued requests.
    
    The current policy is designed to optimize the throughput. First,
    it batches as many prefill requests as possible. And it schedules
    decodes. If there's a pressure on GPU memory, decode requests can
    be swapped or preempted.
    """
    # Include running requests to the budget.
    budget = SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )
    # Make sure we include num running seqs before scheduling prefill,
    # so that we don't schedule beyond max_num_seqs for prefill.
    for seq_group in self.running:
        budget.add_num_seqs(seq_group.request_id,
                            seq_group.get_max_num_running_seqs())
    curr_loras = set(
        seq_group.lora_int_id for seq_group in self.running
        if seq_group.lora_int_id > 0) if self.lora_enabled else None

    prefills = SchedulerPrefillOutputs.create_empty()
    running_scheduled = SchedulerRunningOutputs.create_empty()
    swapped_in = SchedulerSwappedInOutputs.create_empty()

    #import pdb
    #pdb.set_trace()
    # If any requests are swapped, prioritized swapped requests.
    if not self.swapped:
        prefills = self._schedule_prefills(budget,
                                            curr_loras,
                                            enable_chunking=False)

    # Don't schedule decodes if prefills are scheduled.
    # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
    # only contains decode requests, not chunked prefills.
    
    
    if len(prefills.seq_groups) == 0:
        # Jiayi Modification starts
        running_scheduled = self._schedule_running(budget,
                                                    curr_loras,
                                                    enable_chunking=False)
        # Jiayi Modification ends
        
        # If any sequence group is preempted, do not swap in any sequence
        # group. because it means there's no slot for new running requests.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

    assert (budget.num_batched_tokens <=
            self.scheduler_config.max_num_batched_tokens)
    assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

    # Update waiting requests.
    self.waiting.extendleft(running_scheduled.preempted)
    # Update new running requests.
    if len(prefills.seq_groups) > 0:
        self.running.extend([s.seq_group for s in prefills.seq_groups])

    self.running.extend(running_scheduled.decode_seq_groups_list)

    if len(swapped_in.decode_seq_groups) > 0:
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])

    # Update swapped requests.
    self.swapped.extend(running_scheduled.swapped_out)
    preempted = (len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out))

    # There should be no prefill from running queue because this policy
    # doesn't allow chunked prefills.
    assert len(running_scheduled.prefill_seq_groups) == 0
    assert len(swapped_in.prefill_seq_groups) == 0

    # Merge lists
    num_prefill_groups = len(prefills.seq_groups)
    if num_prefill_groups > 0:
        scheduled_seq_groups = prefills.seq_groups
        scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
    else:
        scheduled_seq_groups = running_scheduled.decode_seq_groups
    scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

    blocks_to_copy = running_scheduled.blocks_to_copy
    blocks_to_copy.extend(swapped_in.blocks_to_copy)

    ignored_seq_groups = prefills.ignored_seq_groups
    ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

    return SchedulerOutputs(
        scheduled_seq_groups=scheduled_seq_groups,
        num_prefill_groups=num_prefill_groups,
        num_batched_tokens=budget.num_batched_tokens,
        blocks_to_swap_in=swapped_in.blocks_to_swap_in,
        blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        ignored_seq_groups=ignored_seq_groups,
        num_lookahead_slots=running_scheduled.num_lookahead_slots,
        running_queue_size=len(self.running),
        preempted=preempted,
    )
    
def new_schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
    # Schedule sequence groups.
    # This function call changes the internal states of the scheduler
    # such as self.running, self.swapped, and self.waiting.
    scheduler_start_time = time.perf_counter()

    #import pdb
    #pdb.set_trace()
    scheduler_outputs = self._schedule()
    #pdb.set_trace()
    now = time.time()

    if not self.cache_config.enable_prefix_caching:
        common_computed_block_nums = []

    allow_async_output_proc: bool = self.use_async_output_proc

    # Create input data structures.
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for i, scheduled_seq_group in enumerate(
            scheduler_outputs.scheduled_seq_groups):
        seq_group = scheduled_seq_group.seq_group
        token_chunk_size = scheduled_seq_group.token_chunk_size
        seq_group.maybe_set_first_scheduled_time(now)

        seq_group_metadata = self._seq_group_metadata_cache[
            self.cache_id].get_object()
        seq_group_metadata.seq_data.clear()
        seq_group_metadata.block_tables.clear()

        # seq_id -> SequenceData
        seq_data: Dict[int, SequenceData] = {}
        # seq_id -> physical block numbers
        block_tables: Dict[int, List[int]] = {}

        if seq_group.is_encoder_decoder():
            # Encoder associated with SequenceGroup
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            encoder_seq_data = encoder_seq.data
            # Block table for cross-attention
            # Also managed at SequenceGroup level
            cross_block_table = self.block_manager.get_cross_block_table(
                seq_group)
        else:
            encoder_seq_data = None
            cross_block_table = None

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_id = seq.seq_id
            seq_data[seq_id] = seq.data
            block_tables[seq_id] = self.block_manager.get_block_table(seq)
            self.block_manager.access_all_blocks_in_seq(seq, now)

        if self.cache_config.enable_prefix_caching:
            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))
        
        do_sample = True
        is_prompt = seq_group.is_prefill()
        # We should send the metadata to workers when the first prefill
        # is sent. Subsequent requests could be chunked prefill or decode.
        is_first_prefill = False
        if is_prompt:
            seqs = seq_group.get_seqs()
            # Prefill has only 1 sequence.
            assert len(seqs) == 1
            num_computed_tokens = seqs[0].data.get_num_computed_tokens()
            is_first_prefill = num_computed_tokens == 0
            # In the next iteration, all prompt tokens are not computed.
            # It means the prefill is chunked, and we don't need sampling.
            # NOTE: We use get_len instead of get_prompt_len because when
            # a sequence is preempted, prefill includes previous generated
            # output tokens.
            if (token_chunk_size + num_computed_tokens <
                    seqs[0].data.get_len()):
                do_sample = False

        # It assumes the scheduled_seq_groups is ordered by
        # prefill < decoding.
        if is_first_prefill or not self.scheduler_config.send_delta_data:
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                encoder_seq_data=encoder_seq_data,
                cross_block_table=cross_block_table,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
                prompt_adapter_request=seq_group.prompt_adapter_request,
            )
        else:
            # When SPMD mode is enabled, we only send delta data except for
            # the first request to reduce serialization cost.
            seq_data_delta = {}
            for id, data in seq_data.items():
                seq_data_delta[id] = data.get_delta_and_reset()
            seq_group_metadata = SequenceGroupMetadataDelta(
                seq_data_delta,
                seq_group.request_id,
                block_tables,
                is_prompt,
                do_sample=do_sample,
                token_chunk_size=token_chunk_size,
                computed_block_nums=common_computed_block_nums,
            )
        seq_group_metadata_list.append(seq_group_metadata)

        if allow_async_output_proc:
            allow_async_output_proc = self._allow_async_output_proc(
                seq_group)

    # Now that the batch has been created, we can assume all blocks in the
    # batch will have been computed before the next scheduling invocation.
    # This is because the engine assumes that a failure in model execution
    # will crash the vLLM instance / will not retry.
    for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
        self.block_manager.mark_blocks_as_computed(
            scheduled_seq_group.seq_group,
            scheduled_seq_group.token_chunk_size)

    self._seq_group_metadata_cache[self.next_cache_id].reset()

    scheduler_time = time.perf_counter() - scheduler_start_time
    # Add this to scheduler time to all the sequences that are currently
    # running. This will help estimate if the scheduler is a significant
    # component in the e2e latency.
    for seq_group in self.running:
        if seq_group is not None and seq_group.metrics is not None:
            if seq_group.metrics.scheduler_time is not None:
                seq_group.metrics.scheduler_time += scheduler_time
            else:
                seq_group.metrics.scheduler_time = scheduler_time

    # Move to next cache (if exists)
    self.cache_id = self.next_cache_id

    # Return results
    return (seq_group_metadata_list, scheduler_outputs,
            allow_async_output_proc)