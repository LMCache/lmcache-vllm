import torch
import yaml

from lmcache.cache_engine import LMCacheEngineBuilder
from lmcache_vllm import LMCVLLMDriver_V2#, broadcast_tokens_and_block_tables, broadcast_input_ids_list
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

if __name__ == '__main__':
    
    # TODO(Jiayi): Most configs are hard-coded for now
    # Maybe they can be sent from vllm during init
    comm_config = yaml.safe_load(open("/dataheart/jiayi3/lmcache-2/LMCache/examples/comm_config.yaml"))
    vllm_config = yaml.safe_load(open("/dataheart/jiayi3/lmcache-2/LMCache/examples/vllm_config.yaml"))
    
    lmcache_config_file = "/dataheart/jiayi3/lmcache-2/LMCache/examples/example.yaml"
    
    
    lmcache_metadata = LMCacheEngineMetadata(
        "mistral-7b",#model_name, 
        comm_config.get("world_size"),#maybe_lmc_vllm_world_size, 
        comm_config.get("lmc_rank"),#maybe_vllm_rank, 
        fmt="vllm")
    lmcache_config = LMCacheEngineConfig.from_file(lmcache_config_file)
    cache_engine = LMCacheEngineBuilder.get_or_create("vllm", lmcache_config, lmcache_metadata)
    
    # Init LMCEngine & LMCDriver
    cache_engine = LMCacheEngineBuilder.get("vllm")
    lmcache_driver = LMCVLLMDriver_V2(vllm_config, comm_config, cache_engine)
    
    # Start LMC
    lmcache_driver.run()
    