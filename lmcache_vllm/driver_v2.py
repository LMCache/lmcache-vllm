from typing import List, Tuple, Any
from itertools import chain
import torch
import time

from lmcache.cache_engine import LMCacheEngine
from lmcache.logging import init_logger

#import vllm.distributed.distributed_kv as dist_kv

logger = init_logger(__name__)

# Use this tag for all lmcache/disagg prefill logic
DISTRIBUTED_KV_GLOO_TAG = 24857323

# FIXME(Jiayi): sometimes the kv might be 8-bit while hidden_states is 16-bit



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
        
        # initialize transport layer
        self.transport = Transport(comm_config)
        
        # lmc cache engine
        self.cache_engine = cache_engine
        
        # others
        logger.info("LMCache driver initialized!!!")
        
        
        # Meta
        #meta = {
        #    token_ids: List[torch.Tensor],
        #    layer_range: List[List[Tuple]],
        #    token_range: List[List[Tuple]],
        #    head_range: List[List[Tuple]],
        #}

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
    
    # TODO(Jiayi): we need to break our assumption of
    # decoupling device (simply receiving and retrieving a cuda tensor)
    # maybe a cpu tensor?
    
    def recv_kv_and_store(
        self,
    ):
        while True: 
            # ping vllm to kick off kv cache transfer
            # send input tokens
            self.pipe.send_tensor(None)
            # send roi (TODO(Jiayi): roi can be skipped??)
            self.pipe.send_tensor(None)
            
            input_tokens = self.pipe.recv_tensor()
            if input_tokens is None:
                logger.debug(f"vllm buffer is empty. Nothing has been retrieved...")
                # TODO(Jiayi): need to have some kind of rate control logic
                time.sleep(1)
            roi = self.pipe.recv_tensor()
            
            # kv shape is [num_layer, num_toks, num_heads, head_size]
            keys = self.pipe.recv_tensor()
            values = self.pipe.recv_tensor()
            
            # TODO (Jiayi): Is there a way to skip this in lmcache
            # recv useless hidden_or_intermediate states
            _ = self.pipe.recv_tensor()
                
            keys = torch.unbind(keys)
            values = torch.unbind(values)
            rebuilt_kv_cache = []
            for layer_idx in len(keys):
                rebuilt_kv_cache.append((keys[layer_idx], values[layer_idx]))
            
            self.cache_engine.store(input_tokens, rebuilt_kv_cache, blocking=False)

    


        
    
    def retrive_kv_and_send(
        self,
    ):
        meta_r = self.transport.receive_object()
        token_ids = meta_r["token_ids"]
        
        meta_s = {
            "token_ids": meta_r["token_ids"],
            "layer_range": meta_r["token_ids"],
            "token_range": [],
            "head_range": meta_r["token_ids"]}
        logger.info(f"Retrieving {len(token_ids)} reqs")
        
        # This is inefficient extra temporary buffer when multiple reqs are received
        tuple_kvs = []
        for token_tensor in token_ids:
            tuple_kv, num_computed_tok = self.cache_engine.retrive(token_tensor, self.device)
            # FIXME(Jiayi): Prefix caching is assumed here
            meta_s["token_range"].append((0, num_computed_tok))
            tuple_kvs.append(tuple_kv)
            if num_computed_tok > 0:
                num_hit += 1
        logger.info(f"{num_hit} out of {num_req} reqs are hit")
        
        num_req = len(token_ids)
        for req_idx in num_req:
            
            # skip send if cache not hit
            if not tuple_kv:
                continue
            #tuple_kv: (K,V)*num_layer
            #K/V: [num_retrieved_tokens, num_heads, head_size]  
            
            layer_range = meta["layer_range"][req_idx]
            
            for idx in range(len(layer_range)):
                layer_indices.extend([i for i in range(layer_range[idx][0], layer_range[idx][1])])
            
            for l in layer_indices:
                logger.debug(f"sending layer {l}")
                # send key tensor
                key_tensor = tuple_kv[l][0]
                self.transport.send(key_tensor)
                
                # send value tensor
                value_tensor = tuple_kv[l][1]
                self.transport.send(value_tensor)
            
            # FIXME(Jiayi): need to send intermediate states instead of a tensor
            # send a useless hidden states
            logger.debug(f"sending null hidden states")
            null_hidden_tensor = torch.zeros(
                (len(token_ids), self.hidden_size), 
                dtype=self.dtype) # null hidden tensor
            self.transport.send(null_hidden_tensor)

        
    def run(
        self,
    ):
        # Do we need an end signal in recv
        # Otherwise, we need two separate threads for retrieve_kv and recv_kv
        while True:
            self.retrive_kv_and_send()
            self.recv_kv_and_store()

'''  
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
        
        # initialize transport layer
        self.transport = Transport(comm_config)
        
        # lmc cache engine
        self.cache_engine = cache_engine
        
        # others
        logger.info("LMCache driver initialized!!!")
        
        
        # Meta
        #meta = {
        #    token_ids: List[torch.Tensor],
        #    layer_range: List[List[Tuple]],
        #    token_range: List[List[Tuple]],
        #    head_range: List[List[Tuple]],
        #}

        
       
    def recv_kv_and_store(
        self,
    ):
            
        # ping vllm to kick off kv cache transfer
        null_meta = {
            "token_ids": -1,
        }
        self.transport.send_object(null_meta)
        
        # get metadata from vllm
        meta = self.transport.recv_object()
        num_req = len(meta["token_ids"])
        logger.debug(f"receiving request...")
        
        for req_idx in range(num_req):
            
            # FIXME(Jiayi): need to put all info (e.g., layer indices) into lmcache key
            token_tensor = meta["token_ids"][req_idx]
            layer_range = meta["layer_range"][req_idx]
            head_range = meta["head_range"][req_idx]
            layer_indices = []
            head_indices = []
            for idx in range(len(layer_range)):
                layer_indices.extend([i for i in range(layer_range[idx][0], layer_range[idx][1])])
                head_indices.extend([i for i in range(head_range[idx][0], head_range[idx][1])])
            # Here, we assume the KV of all tokens are recieved
            num_tok = len(token_tensor)
            
            kv_size_per_layer = (num_tok, len(head_indices), self.head_size)
            
            rebuilt_kv_cache = []
            for l in layer_indices:
                logger.debug(f"receiving layer {l}")
                
                # receive key tensor
                key_tensor = self.transport.recv(kv_size_per_layer, dtype=self.dtype)
                    
                # receive value tensor
                value_tensor = self.transport.recv(kv_size_per_layer, dtype=self.dtype)
                
                rebuilt_kv_cache.append((key_tensor, value_tensor))
            
            self.cache_engine.store(token_tensor, rebuilt_kv_cache, blocking = False)
            
            # TODO(Jiayi): Is there a way to simply skip receiving `hidden_states`
            null_hidden_size = (num_tok, self.hidden_size)
            null_hidden_states = self.transport.recv(null_hidden_size, dtype=self.dtype)
    


        
    
    def retrive_kv_and_send(
        self,
    ):
        meta_r = self.transport.receive_object()
        token_ids = meta_r["token_ids"]
        
        meta_s = {
            "token_ids": meta_r["token_ids"],
            "layer_range": meta_r["token_ids"],
            "token_range": [],
            "head_range": meta_r["token_ids"]}
        logger.info(f"Retrieving {len(token_ids)} reqs")
        
        # This is inefficient extra temporary buffer when multiple reqs are received
        tuple_kvs = []
        for token_tensor in token_ids:
            tuple_kv, num_computed_tok = self.cache_engine.retrive(token_tensor, self.device)
            # FIXME(Jiayi): Prefix caching is assumed here
            meta_s["token_range"].append((0, num_computed_tok))
            tuple_kvs.append(tuple_kv)
            if num_computed_tok > 0:
                num_hit += 1
        logger.info(f"{num_hit} out of {num_req} reqs are hit")
        
        num_req = len(token_ids)
        for req_idx in num_req:
            
            # skip send if cache not hit
            if not tuple_kv:
                continue
            #tuple_kv: (K,V)*num_layer
            #K/V: [num_retrieved_tokens, num_heads, head_size]  
            
            layer_range = meta["layer_range"][req_idx]
            
            for idx in range(len(layer_range)):
                layer_indices.extend([i for i in range(layer_range[idx][0], layer_range[idx][1])])
            
            for l in layer_indices:
                logger.debug(f"sending layer {l}")
                # send key tensor
                key_tensor = tuple_kv[l][0]
                self.transport.send(key_tensor)
                
                # send value tensor
                value_tensor = tuple_kv[l][1]
                self.transport.send(value_tensor)
            
            # FIXME(Jiayi): need to send intermediate states instead of a tensor
            # send a useless hidden states
            logger.debug(f"sending null hidden states")
            null_hidden_tensor = torch.zeros(
                (len(token_ids), self.hidden_size), 
                dtype=self.dtype) # null hidden tensor
            self.transport.send(null_hidden_tensor)

        
    
    def run(
        self,
    ):
        # Do we need an end signal in recv
        # Otherwise, we need two separate threads for retrieve_kv and recv_kv
        while True:
            self.retrive_kv_and_send()
            self.recv_kv_and_store()
'''