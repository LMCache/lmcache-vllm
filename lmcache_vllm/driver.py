from typing import List, Tuple
import torch

from vllm.attention import AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
import vllm._custom_ops as ops
from vllm.config import ModelConfig, ParallelConfig
from vllm.sequence import SequenceGroupMetadata

from lmcache.cache_engine import LMCacheEngine
from lmcache.logging import init_logger

logger = init_logger(__name__)

class LMCVLLMDriver:
    def __init__(
            self, 
            cache_engine: LMCacheEngine, 
            model_config: ModelConfig, 
            parallel_config: ParallelConfig,
            device
        ):
        """
        Initialize the driver

        Arguments:
            cache_engine: the lmcache engine
            model_config: the model_config object from vllm
            parallel_config: the parallel_config object from vllm
            device: the device of the vllm model runner
        """
        self.cache_engine = cache_engine
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device = device

        self.block_size: int

    def set_block_size(
            self, 
            block_size: int,
        ):
        """
        Set the block size 
        Will be called when model_runner.set_block_size() is called
        """
        self.block_size = block_size

    def _inject_kv_cache(
            self,
            kv_caches: List[torch.Tensor],
            loaded_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
            loaded_cache_len: int,
            block_table: List[int]
    ) -> List[int]:
        """
        For cache engine: inject the loaded cache into the kv_caches buffer
        Input:
            kv_caches: the vLLM's KV cache buffer
            loaded_cache: the KV cache loaded from the cache engine.
                          the shape of a single layer's K/V tensor is: [num_tokens, num_heads, head_size]
            block_table: the block table of the corresponding sequence
        Output:
            loaded_block_nums: the block idx of the blocks that are being injected
        """
        slot_mapping = []
        ''' prepare slot_mapping '''
        loaded_block_nums = [block_table[i // self.block_size] for i in range(0, loaded_cache_len, self.block_size)]

        # TODO: use fast tensor operations to generate slot_mapping
        for i in range(loaded_cache_len):
            block_number = block_table[i // self.block_size]
            block_offset = i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)

        # FIXME: what if the model is not on GPU? Consider using to(self.device) instead of cuda()
        slot_mapping = torch.tensor(slot_mapping).cuda()

        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        for gpu_cache_layer, kv in zip(kv_caches, loaded_cache):
            ''' reshape kv to [num tokens, num heads, head size] '''
            k, v = kv
            k_cache, v_cache = PagedAttention.split_kv_cache(gpu_cache_layer, num_kv_heads, head_size)
            ops.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping, "auto", 1.0)
        logger.info(f"Injected {len(loaded_block_nums) - 1} blocks")
        return loaded_block_nums[:-1]

    @torch.no_grad()
    def retrive_and_inject(
            self,
            kv_caches: List[torch.Tensor],
            token_ids: List[int],
            block_table: List[int],
    ) -> List[int]:
        """
        For cache engine: inject the loaded cache into the kv_caches buffer
        Input:
            kv_caches: the vLLM's KV cache buffer
            tokens_ids: a list of ints representing the token ids
            block_table: the block table of the corresponding sequence
        Output:
            loaded_block_nums: the block idx of the blocks that are being injected.
                               Can be an empty list if nothing is injected
        """
        loaded_kv, loaded_kv_len = self.cache_engine.retrive(torch.tensor(token_ids), self.device)
        if loaded_kv_len > self.block_size: # skip if less than a single block
            loaded_block_nums = self._inject_kv_cache(kv_caches, loaded_kv, loaded_kv_len, block_table)
            return loaded_block_nums
        else:
            return []

    @torch.no_grad()
    def collect_kv_and_store(
            self,
            kv_caches: List[torch.Tensor],
            token_ids: torch.Tensor,
            attn_metadata: AttentionMetadata,
            idx: int
        ) -> None:
        """
        For lmcache engine: put the paged KV cache together and store it into the cache engine
        Input:
            kv_caches: the vLLM's KV cache buffer
            tokens_ids: a 1D tensor of ints representing the token ids
            attn_metadata: the attention metadata
            idx: the index in the current batch
        """
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        slot_mapping = torch.zeros_like(token_ids).to(self.device)

        # prepare slot mapping
        block_table = attn_metadata.prefill_metadata.block_tables[idx]
        context_length = attn_metadata.prefill_metadata.context_lens
        num_blocks = len(block_table)
        prefix_slot_mapping = block_table.repeat_interleave(self.block_size) * self.block_size + \
                torch.arange(self.block_size).repeat(num_blocks).to(block_table.device)
        slot_mapping[:context_length] = prefix_slot_mapping[:context_length]
        st, ed = attn_metadata.prefill_metadata.subquery_start_loc[idx:idx+2]
        slot_mapping[context_length:] = attn_metadata.slot_mapping[st:ed]

        rebuilt_kv_cache = []
        # FIXME: the following code is not really readable
        # FIXME(Jiayi): a load kernel could make the following code faster
        for kv_layer in kv_caches:
            k_cache, v_cache = PagedAttention.split_kv_cache(kv_layer, num_kv_heads, head_size)
            v = v_cache.permute([0, 3, 1, 2]).reshape(-1, num_kv_heads, head_size)[slot_mapping].contiguous()
            k = k_cache.permute([0, 3, 1, 2, 4]).reshape(-1, num_kv_heads, head_size)[slot_mapping].contiguous()
            rebuilt_kv_cache.append((k, v))

        self.cache_engine.store(token_ids, rebuilt_kv_cache, blocking = False)

    def retrive(
            self,
            kv_caches: List[torch.Tensor],
            seq_group_metadata: SequenceGroupMetadata,
        ) -> List[int]:
        """
        Given a sequence group, this function tries to retrive the KV cache from the cache engine
        and injects it into vLLM's kv cache object

        Inputs:
            kv_caches: the vLLM's KV cache buffer
            seq_group_metadata: the SequenceGroupMetadata object for the sequence group

        Returns:
            loaded_block_nums: the block idx of the blocks that are being injected.
                               Can be an empty list if nothing is injected
        """
        loaded_block_nums = []
        seq_ids = seq_group_metadata.seq_data.keys()
        seq_id = list(seq_ids)[0]
        seq_data = seq_group_metadata.seq_data[seq_id]
        if self.cache_engine is not None and seq_group_metadata.block_tables is not None:
            block_table = seq_group_metadata.block_tables[seq_id]
            loaded_block_nums = self.retrive_and_inject(kv_caches, seq_data.get_token_ids(), block_table)
        return loaded_block_nums

