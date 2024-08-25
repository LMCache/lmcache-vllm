from typing import List, Tuple, Any
import torch
'''
from vllm.attention import AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
import vllm._custom_ops as ops
from vllm.config import ModelConfig, ParallelConfig
from vllm.sequence import SequenceGroupMetadata
'''
from lmcache.cache_engine import LMCacheEngine
from lmcache.logging import init_logger

#import vllm.distributed.distributed_kv as dist_kv

logger = init_logger(__name__)

# Use this tag for all lmcache/disagg prefill logic
DISTRIBUTED_KV_GLOO_TAG = 24857323

class LMCVLLMDriver_V2:
    def __init__(
        self,
        vllm_config,
        comm_config,
        cache_engine,
    ):
        # vllm-related configs
        self.start_layer = vllm_config.get("start_layer")
        self.end_layer = vllm_config.get("end_layer")
        self.num_layer = self.end_layer - self.start_layer
        self.num_heads = vllm_config.get("num_heads")
        self.head_size = vllm_config.get("head_size")
        self.dtype = vllm_config.get("dtype")
        if self.dtype == "float16":
            self.dtype = torch.float16
        self.hidden_size = vllm_config.get("hidden_size")
        
        # communication configs
        self.backend = comm_config.get("backend")
        self.world_size =comm_config.get("world_size")
        self.lmc_rank =comm_config.get("lmc_rank")
        self.distributed_init_method = comm_config.get("distributed_init_method")
        self.target_rank_for_recv = comm_config.get("target_rank_for_recv")
        self.target_rank_for_send = comm_config.get("target_rank_for_send")
        self.device = torch.device("cpu")
        torch.distributed.init_process_group(
            backend=self.backend,
            init_method=self.distributed_init_method,
            world_size=self.world_size,
            rank=self.lmc_rank)
        
        # FIXME(Jiayi): remove this hardcode
        ranks = [0]
        self.device_group_world = torch.distributed.new_group(ranks, backend=self.backend)
        self.cpu_group_world = torch.distributed.new_group(ranks, backend="gloo")
        
        self.device_group_TP = torch.distributed.new_group(ranks, backend=self.backend)
        self.cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")
        
        self.device_group_PP = torch.distributed.new_group(ranks, backend=self.backend)
        self.cpu_group_PP = torch.distributed.new_group(ranks, backend="gloo")
        
        # FIXME(Jiayi): remove this hardcode
        ranks = [0, 1]
        self.device_group = torch.distributed.new_group(ranks, backend=self.backend)
        self.cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        
        # lmc cache engine
        self.cache_engine = cache_engine
        
        # others

        logger.info("LMCache driver initialized!!!")
    
    def send_object(self, obj):

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size
        torch.distributed.send(size_tensor,
                               dst=self.target_rank_for_send, 
                               group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=self.target_rank_for_send, 
                               group=self.cpu_group)

        return None

    def recv_object(self) -> Any:

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor,
                                           src=self.target_rank_for_recv,
                                           group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu")

        rank_object = torch.distributed.recv(object_tensor,
                                             src=self.target_rank_for_recv,
                                             group=self.cpu_group)

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    
    def recv_kv_start(
        self,
    ):
        
        null_size_tensor = torch.tensor([1,1])
        torch.distributed.send(
            null_size_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Sending null size tensor done")
        
        null_token_ids_list_tensor = torch.tensor([[-1]])
        torch.distributed.send(
            null_token_ids_list_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Sending null token_ids_list tensor done")
    

        
       
    def recv_kv_and_store(
        self,
    ):
            
        # ping vllm to kick off kv cache transfer
        self.recv_kv_start()
        
        
        while True:
            logger.debug(f"receiving request...")
            
            #FIXME(Jiayi): missing an end signal handler here
            # Receive num computed token tensor
            num_computed_token_tensor = torch.tensor([0])
            torch.distributed.recv(
                num_computed_token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            num_computed_token = num_computed_token_tensor.item()
            
            # handling end signal
            if num_computed_token == -1:
                logger.debug(f"received end signal, exiting recv")
                break
            
            # Receive num token tensor
            num_token_tensor = torch.tensor([0])
            torch.distributed.recv(
                num_token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            num_token = num_token_tensor.item()
            
            # Receive token tensor
            token_tensor = torch.empty((num_token,), dtype=torch.long)
            torch.distributed.recv(
                token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            #print(token_tensor)
            #token_ids = token_tensor.tolist()
            
            rebuilt_kv_cache = []
            for l in range(self.num_layer):
                logger.debug(f"receiving layer {l}")
                
                # receive key tensor
                key_tensor = torch.empty((num_computed_token, self.num_heads, self.head_size),
                                        dtype=self.dtype)
                torch.distributed.recv(
                    key_tensor,
                    self.target_rank_for_recv,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
                    
                # receive value tensor
                value_tensor = torch.empty((num_computed_token, self.num_heads, self.head_size),
                                        dtype=self.dtype)
                torch.distributed.recv(
                    value_tensor,
                    self.target_rank_for_recv,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
                
                rebuilt_kv_cache.append((key_tensor, value_tensor))
            
            self.cache_engine.store(token_tensor, rebuilt_kv_cache, blocking = False)
            
            # TODO(Jiayi): Is there a way to simply skip receiving `hidden_states`
            null_hidden_states = torch.empty([num_token, self.hidden_size],
                                        dtype=self.dtype)
            torch.distributed.recv(
                null_hidden_states,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
    

    def send_kv_start(
        self,
    ):
        
        size_tensor = torch.tensor([0,0])
        torch.distributed.recv(
            size_tensor,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Receiving size tensor done")
        
        token_ids_list_tensor = torch.empty(size_tensor.tolist(), dtype=torch.long)
        torch.distributed.recv(
            token_ids_list_tensor,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Receiving token_ids_list tensor done")
        return token_ids_list_tensor.tolist()

        
    
    # send is not muti-threaded for now
    def retrive_kv_and_send(
        self,
    ):
        token_ids_list = self.send_kv_start()
        logger.info(f"Retrieving {len(token_ids_list)} reqs")
        num_req = len(token_ids_list)
        num_hit = 0
        for token_ids in token_ids_list:
            
            #tuple_kv: (K,V)*num_layer
            #K/V: [num_retrieved_tokens, num_heads, head_size]
            token_tensor = torch.tensor(token_ids, device=self.device)
            num_tok = len(token_ids)
            tuple_kv, num_computed_tok = self.cache_engine.retrive(token_tensor, self.device)
            
            # send num_computed_tok
            num_computed_tok_tensor = torch.tensor([num_computed_tok], device=self.device)
            torch.distributed.send(
                num_computed_tok_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            # send num_tok
            num_tok_tensor = torch.tensor([num_tok], device=self.device)
            torch.distributed.send(
                num_tok_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            # send token_ids back to vllm
            torch.distributed.send(
                token_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            if num_computed_tok == 0:
                continue
            
            
            for l in range(self.num_layer):
                logger.debug(f"sending layer {l}")
                # send key tensor
                key_tensor = tuple_kv[l][0]
                torch.distributed.send(
                    key_tensor,
                    self.target_rank_for_send,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
                
                # send value tensor
                value_tensor = tuple_kv[l][1]
                torch.distributed.send(
                    value_tensor,
                    self.target_rank_for_send,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
            
            # FIXME(Jiayi): need to send intermediate states instead of a tensor
            # send a useless hidden states
            logger.debug(f"sending hidden states")
            null_hidden_tensor = torch.zeros(
                (len(token_ids), self.hidden_size), 
                dtype=self.dtype) # null hidden tensor
            torch.distributed.send(
                null_hidden_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            num_hit += 1
            
        # Send end signal
        logger.debug(f"sending end signal")
        end_signal_tensor = torch.tensor([-1])
        torch.distributed.send(
            end_signal_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        
        logger.info(f"{num_hit} out of {num_req} reqs are hit")
    
    def run(
        self,
    ):
        while True:
            self.retrive_kv_and_send()
            self.recv_kv_and_store()
        
    
'''
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
        #prepare slot_mapping
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
            # reshape kv to [num tokens, num heads, head size]
            k, v = kv
            k_cache, v_cache = PagedAttention.split_kv_cache(gpu_cache_layer, num_kv_heads, head_size)
            ops.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping, "auto", 1.0)
        logger.info(f"Injected {len(loaded_block_nums) - 1} blocks")
        return loaded_block_nums[:-1]

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
        #rebuilt_k_caches = []
        #rebuilt_v_caches = []
        # FIXME: the following code is not really readable
        for kv_layer in kv_caches:
            k_cache, v_cache = PagedAttention.split_kv_cache(kv_layer, num_kv_heads, head_size)
            v = v_cache.permute([0, 3, 1, 2]).reshape(-1, num_kv_heads, head_size)[slot_mapping]
            k = k_cache.permute([0, 3, 1, 2, 4]).reshape(-1, num_kv_heads, head_size)[slot_mapping]
            rebuilt_kv_cache.append((k, v))
            
            #rebuilt_k_caches.append(k)
            #rebuilt_v_caches.append(v)
        #rebuilt_k_cache = torch.stack(rebuilt_k_caches)
        #rebuilt_v_cache = torch.stack(rebuilt_v_caches)
        
        # rebuilt_kv_cache: [num_layer, 2, num_tok, num_kv_head, head_size]
        #rebuilt_kv_cache = torch.stack((rebuilt_k_cache, rebuilt_v_cache))
        #rebuilt_kv_cache = rebuilt_kv_cache.permute([1, 0, 2, 3, 4])
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
'''
