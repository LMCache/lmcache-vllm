from typing import List, Tuple, Any
from itertools import chain
import torch
import time
import threading

from lmcache.cache_engine import LMCacheEngine
from lmcache.logging import init_logger
from .utils import init_comm
from .pipe import TorchDistributedPipe

#import vllm.distributed.distributed_kv as dist_kv

logger = init_logger(__name__)

# Use this tag for all lmcache/disagg prefill logic
DISTRIBUTED_KV_GLOO_TAG = 24857323

# FIXME(Jiayi): sometimes the kv might be 8-bit while hidden_states is 16-bit
class LMCVLLMDriver_V2:
    def __init__(
        self,
        vllm_config,
        lmc_rank: int,
        group_ranks: List[List[int]],
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
        
        # comm related configs
        # TODO (Jiayi): simplify the logic and remove hardcodes
        backend = vllm_config.get("backend")
        tensor_model_parallel_size = vllm_config.get("tensor_model_parallel_size")
        pipeline_model_parallel_size = vllm_config.get("pipeline_model_parallel_size")
        distributed_init_method = vllm_config.get("distributed_init_method")
        
        # Init communications and crate pipes
        pipes = init_comm(
            backend, 
            lmc_rank, 
            group_ranks,
            tensor_model_parallel_size, 
            pipeline_model_parallel_size, 
            distributed_init_method)
        
        self.recv_pipe, self.recv_signal_pipe, self.send_pipe, self.send_signal_pipe = \
            pipes[0], pipes[1], pipes[2], pipes[3]
        
        
        # lmc cache engine
        self.cache_engine = cache_engine
        # HACK(Jiayi): this is curently a hack
        # might error in multi-layered local backend
        cache_engine.engine_.dst_device = "cpu"
        
        # others
        logger.info("LMCache driver initialized!!!")
        
        # Indicate signals
        self.normal_signal = torch.tensor([0])
        self.end_signal = None
        
        # Start recv and send threads
        self.send_thread = threading.Thread(target=self.retrive_kv_and_send, args=())
        self.recv_thread = threading.Thread(target=self.recv_kv_and_store, args=())
        

        # Protocol
        # Send-------------------------------Recv
        # <<<<<<<<<<<<<<<<<<<<<<<<<< input_tokens
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< roi
        # input_tokens >>>>>>>>>>>>>>>>>>>>>>>>>>
        # roi >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # keys >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # hidden >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        #[input_tokens, roi, key, value, hidden]
    
    
    def recv_kv_and_store(
        self,
    ):
        while True: 
            # ping vllm to kick off kv cache transfer
            # send input tokens
            self.recv_signal_pipe.send_tensor(self.normal_signal)
            self.recv_pipe.send_tensor(None)
            # send roi (TODO(Jiayi): roi can be skipped??)
            self.recv_pipe.send_tensor(None)
            
            input_tokens = self.recv_pipe.recv_tensor()
            if input_tokens is None:
                logger.debug(f"vllm buffer is empty. Nothing has been retrieved...")
                # TODO(Jiayi): need to have some kind of rate control logic here
                time.sleep(1)
                continue
            # recv redundant roi
            _ = self.recv_pipe.recv_tensor()
            
            # kv shape is [num_layer, num_toks, num_heads, head_size]
            keys = self.recv_pipe.recv_tensor()
            values = self.recv_pipe.recv_tensor()
            
            # TODO (Jiayi): Is there a way to skip this in lmcache
            # recv useless hidden_or_intermediate states
            _ = self.recv_pipe.recv_tensor()
            
            # TODO (Jiayi): unbind can be optimized by changing cache_engine 
            keys = torch.unbind(keys)
            values = torch.unbind(values)
            rebuilt_kv_cache = []
            for layer_idx in range(len(keys)):
                #logger.debug(f"received k shape {keys[layer_idx].shape}")
                rebuilt_kv_cache.append((keys[layer_idx], values[layer_idx]))
            
            self.cache_engine.store(input_tokens, rebuilt_kv_cache, blocking=False)

    
    def _is_end_signal(self, signal):
        return signal is None
    
    def retrive_kv_and_send(
        self,
    ):
        while True:
            signal = self.send_signal_pipe.recv_tensor()
            logger.debug(f"Received signal {signal} in retrive_kv_and_send")
            if self._is_end_signal(signal):
                logger.info("Received end signal!")
                break
            input_tokens = self.send_pipe.recv_tensor()
            roi_null = self.send_pipe.recv_tensor()
            
            # assume vllm wants kv cache of all tokens
            assert len(roi_null) == len(input_tokens)
            
            # TODO(Jiayi): retrieve needs to put tensor on cpu
            tuple_kv, num_computed_tok = self.cache_engine.retrive(input_tokens)
            if num_computed_tok==0:
                self.send_pipe.send_tensor(None) # null input_tensor
                #logger.debug(f"Sent null input tokens in retrive_kv_and_send")
                
                # TODO(Jiayi): the following sends ca be optimized w.
                # an earlier None handler on vllm side
                self.send_pipe.send_tensor(None) # null roi
                self.send_pipe.send_tensor(None) # null key
                self.send_pipe.send_tensor(None) # null value
                self.send_pipe.send_tensor(None) # null hidden
                continue
            
            # TODO (Jiayi): The following loop and cat can be optimized by changing cache_engine
            key_list = []
            value_list = []
            for layer_idx in range(len(tuple_kv)):
                key_list.append(torch.unsqueeze(tuple_kv[layer_idx][0], dim=0))
                value_list.append(torch.unsqueeze(tuple_kv[layer_idx][1], dim=0))
            key = torch.cat(key_list)
            value = torch.cat(value_list)
            roi = torch.tensor([i for i in range(num_computed_tok)])
            
            self.send_pipe.send_tensor(input_tokens) # input_tokens
            self.send_pipe.send_tensor(roi) # roi
            self.send_pipe.send_tensor(key) # key
            self.send_pipe.send_tensor(value) # value
            self.send_pipe.send_tensor(None) # null hdden

    def run(
        self,
    ):
        self.send_thread.start()
        logger.info("LMCache send thread start running!!!")
        self.recv_thread.start()
        logger.info("LMCache recv thread start running!!!")
        
        while True:
            time.sleep(10)
        