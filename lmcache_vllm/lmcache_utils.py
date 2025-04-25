import os
from lmcache.logging import init_logger

if os.getenv("LMCACHE_USE_EXPERIMENTAL") == "True":
    from lmcache.experimental.config import LMCacheEngineConfig
else:
    from lmcache.config import LMCacheEngineConfig

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"

def lmcache_get_config() -> LMCacheEngineConfig:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """
    
    if hasattr(lmcache_get_config, "cached_config"):
        return lmcache_get_config.cached_config

    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn("No LMCache configuration file is set. Trying to read"
                    " configurations from the environment variables.")
        logger.warn("You can set the configuration file through "
                    "the environment variable: LMCACHE_CONFIG_FILE")
        
        # Handle the case where from_env is not available in newer versions
        try:
            config = LMCacheEngineConfig.from_env()
        except AttributeError:
            logger.warn("LMCacheEngineConfig.from_env is not available, using default config")
            # Create a default config with all required parameters
            config = LMCacheEngineConfig(
                chunk_size=512,
                local_device="cuda:0",
                max_local_cache_size=10 * 1024 * 1024,  # 10MB
                remote_url="",
                remote_serde="",
                pipelined_backend=False,
                save_decode_cache=False,
                enable_blending=False,
                blend_recompute_ratio=0.5,
                blend_min_tokens=4
            )
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        try:
            config = LMCacheEngineConfig.from_file(config_file)
        except AttributeError:
            logger.warn("LMCacheEngineConfig.from_file is not available, using default config")
            config = LMCacheEngineConfig(
                chunk_size=512,
                local_device="cuda:0",
                max_local_cache_size=10 * 1024 * 1024,  # 10MB
                remote_url="",
                remote_serde="",
                pipelined_backend=False,
                save_decode_cache=False,
                enable_blending=False,
                blend_recompute_ratio=0.5,
                blend_min_tokens=4
            )

    lmcache_get_config.cached_config = config
    return config
