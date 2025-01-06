import os
from lmcache.config import LMCacheEngineConfig
from lmcache.logging import init_logger

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
        logger.warn("No LMCache configuration file is set. Returning default config")
        logger.warn("Please set the configuration file through "
                    "the environment variable: LMCACHE_CONFIG_FILE")
        config = LMCacheEngineConfig.from_defaults(
                local_device = "cpu",
                remote_url = None,
                remote_serde = None,
                pipelined_backend = False)
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    lmcache_get_config.cached_config = config
    return config