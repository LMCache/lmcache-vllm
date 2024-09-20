from typing import List, Tuple, Any
import torch

from lmcache.logging import init_logger
from .pipe import TorchDistributedPipe
logger = init_logger(__name__)


def init_comm(
    backend: str,
    lmc_rank: int,
    group_ranks: List[List[int]],
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    distributed_init_method: str,
):
    '''
    Initialize both lmcache-related and vllm-related communications here
    '''
    world_size = 2 * tensor_model_parallel_size * pipeline_model_parallel_size
    logger.info(f"Initialize lmc instance (rank {lmc_rank}) with {distributed_init_method}")
    
    torch.distributed.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=lmc_rank)

    logger.info(f"[rank{lmc_rank}]: World initialized")
    
    init_vllm_comm(
        backend, 
        tensor_model_parallel_size, 
        pipeline_model_parallel_size,
        distributed_init_method)
    logger.info("vllm successfully initialized on lmc side")

    # initialize four pipes
    recv_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    recv_signal_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    logger.info("LMCache recv pipe initialized!!!")
    
    send_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    send_signal_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    logger.info("LMCache send pipe initialized!!!")
    
    return recv_pipe, recv_signal_pipe, send_pipe, send_signal_pipe
    

def include_lmcache_groups(
    groups: List[List[int]],
    world_size: int,
) -> List[List[int]]:
    """
        match the distributed group in vLLM instances
        
        vLLM will augment distributed groups when distributed KV transfer is
        enabled
    """
    
    new_groups = []
    for group in groups:
        new_groups.append([rank for rank in group])
    for group in groups:
        new_groups.append([rank + world_size for rank in group])
    return new_groups

    
# TODO (Jiayi): distributed_init_method should be determined in the same way as vllm
def init_vllm_comm(
    backend: str,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    distributed_init_method: str,
):
    '''
    Initialize only vllm-related communications here
    This is needed as any communication group in torch.distributed requires global initialization
    '''
    # each model shard in vllm corresponds to a node
    vllm_world_size = tensor_model_parallel_size * pipeline_model_parallel_size
    
    # Initialize vllm world group
    vllm_ranks = [[i for i in range(vllm_world_size)]]
    vllm_ranks = include_lmcache_groups(vllm_ranks, vllm_world_size)
    for ranks in vllm_ranks:
        device_group_world = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_world = torch.distributed.new_group(ranks, backend="gloo")
    
    # Initialize tp ranks
    tp_ranks = []
    num_tp_groups = (vllm_world_size // tensor_model_parallel_size)
    for i in range(num_tp_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                (i + 1) * tensor_model_parallel_size))
        tp_ranks.append(ranks)
    tp_ranks = include_lmcache_groups(tp_ranks, vllm_world_size)
    for ranks in tp_ranks:
        device_group_TP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")
    
    # Initialize pp ranks
    pp_ranks = []
    num_pp_groups = (vllm_world_size // pipeline_model_parallel_size)
    for i in range(num_pp_groups):
        ranks = list(range(i, vllm_world_size, num_pp_groups))
        pp_ranks.append(ranks)
    pp_ranks = include_lmcache_groups(pp_ranks, vllm_world_size)
    for ranks in pp_ranks:
        device_group_TP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")

