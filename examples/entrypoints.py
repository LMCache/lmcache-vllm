import torch
import yaml
import os
from multiprocessing import Process
import multiprocessing as mp

from lmcache.cache_engine import LMCacheEngineBuilder
from lmcache_vllm import LMCVLLMDriver_V2#, broadcast_tokens_and_block_tables, broadcast_input_ids_list
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

def init_lmc(
    vllm_config,
    lmcache_config_file,
    vllm_rank: int,
    vllm_world_size: int):
    
    # create vllm-lmc pairs
    group_ranks = []
    for i in range(vllm_world_size):
        # vllm rank: i
        # lmc rank: i + world_size
        group_ranks.append([i, i + vllm_world_size])
    
    # TODO(Jiayi): need to use multiprocessing to launch multiple cache engines
    lmc_rank = vllm_rank + vllm_world_size
    
    # FIXME(Jiayi): using `vllm_world_size` is incorrect
    # e.g., tp(2)*pp(4) = tp(1)*pp(8)
    lmcache_metadata = LMCacheEngineMetadata(
        vllm_config.get('model_name'),
        vllm_world_size,
        vllm_rank,
        fmt="vllm")
    lmcache_config = LMCacheEngineConfig.from_file(lmcache_config_file)
    cache_engine = LMCacheEngineBuilder.get_or_create("vllm", lmcache_config, lmcache_metadata)
    
    # Init LMCEngine & LMCDriver
    cache_engine = LMCacheEngineBuilder.get("vllm")
    lmcache_driver = LMCVLLMDriver_V2(vllm_config, lmc_rank, group_ranks, cache_engine)
    
    # Start LMC
    lmcache_driver.run()

    
def get_config_from_env():
    
    configs = {}

    assert "VLLM_HOST_IP" in os.environ and "VLLM_PORT" in os.environ
    ip = os.environ["VLLM_HOST_IP"]
    port = os.environ['VLLM_PORT']
    configs["distributed_init_method"] = f"tcp://{ip}:{port}"

    assert "VLLM_TP" in os.environ
    configs["tp"] = int(os.environ["VLLM_TP"])

    assert "VLLM_PP" in os.environ
    configs["pp"] = int(os.environ["VLLM_PP"])

    assert "VLLM_MODEL_NAME" in os.environ
    configs["model_name"] = os.environ["VLLM_MODEL_NAME"]

    configs["backend"] = "nccl"
    

    return configs
        
    


if __name__ == '__main__':
    # Process has to be spawned insted of forked when using cuda/multi-threading in multiprocessing
    # Otherwise: `RuntimeError: Cannot re-initialize CUDA in forked subprocess``
    mp.set_start_method("spawn")
    
    # TODO(Jiayi): Most configs are hard-coded in yaml for now
    # Maybe they can be sent from vllm during init
    vllm_config = get_config_from_env()
    lmcache_config_file = "lmcache_config.yaml"
    
    tp = vllm_config.get("tp")
    pp = vllm_config.get("pp")
    vllm_world_size = tp * pp

    for vllm_rank in range(vllm_world_size):
        p = Process(target=init_lmc, args=(vllm_config, lmcache_config_file, vllm_rank, vllm_world_size))
        p.start()
    
    #TODO(Jiayi): add error handling and join process