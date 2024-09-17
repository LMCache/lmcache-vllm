import torch
import yaml

from lmcache.cache_engine import LMCacheEngineBuilder
from lmcache_vllm import LMCVLLMDriver_V2#, broadcast_tokens_and_block_tables, broadcast_input_ids_list
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

if __name__ == '__main__':
    
    # TODO(Jiayi): Most configs are hard-coded in yaml for now
    # Maybe they can be sent from vllm during init
    vllm_config = yaml.safe_load(open("vllm_config.yaml"))
    
    lmcache_config_file = "lmcache_config.yaml"
    
    # TODO(Jiayi): need to use multiprocessing to launch multiple cache engines
    vllm_rank = 0
    vllm_world_rank = 1
    lmc_rank = 1
    
    
    lmcache_metadata = LMCacheEngineMetadata(
        "mistral-7b",
        vllm_world_rank,
        vllm_rank,
        fmt="vllm")
    lmcache_config = LMCacheEngineConfig.from_file(lmcache_config_file)
    cache_engine = LMCacheEngineBuilder.get_or_create("vllm", lmcache_config, lmcache_metadata)
    
    # Init LMCEngine & LMCDriver
    cache_engine = LMCacheEngineBuilder.get("vllm")
    lmcache_driver = LMCVLLMDriver_V2(vllm_config, lmc_rank, cache_engine)
    
    # Start LMC
    lmcache_driver.run()
    