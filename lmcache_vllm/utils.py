from typing import List, Tuple, Any
import torch
#from vllm.distributed import broadcast_object_list, broadcast
#from vllm.sequence import SequenceGroupMetadata

'''
def broadcast_list(
        is_driver_worker: bool,
        lis: List[Any],
        device
    ) -> List[Any]:
    """
    Use vllm broadcast primitives to broadcast a list of object from driver worker to other workers
    """
    list_length_tensor = torch.tensor(len(lis), device=device)
    broadcast(list_length_tensor, src = 0)
    list_len = list_length_tensor.item()

    if list_len == 0:
        return []

    if not is_driver_worker:
        lis = [0] * list_len

    broadcast_object_list(lis, src = 0)
    return lis

def broadcast_input_ids_list(
        is_driver_worker: bool,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        device,
    ) -> List[torch.Tensor]:
    """
    Extract a list of input ids from sequence group metadata and broadcast it to other workers
    Each input ids is a integer tensor
    """
    input_ids_list = []
    if is_driver_worker:
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = seq_group_metadata.seq_data.keys()
            assert len(seq_ids) == 1
            seq_id = list(seq_ids)[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            whole_input_tokens = torch.tensor(seq_data.get_token_ids())
            input_ids_list.append(whole_input_tokens)
        broadcast_list(is_driver_worker, input_ids_list, device)
    else:
        input_ids_list = broadcast_list(is_driver_worker, input_ids_list, device)

    return input_ids_list

def broadcast_tokens_and_block_tables(
        is_driver_worker: bool,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        device,
    ) -> List[Tuple[List[int], List[int]]]:
    """
    Extract a list of input ids and block tables from sequence group metadata and broadcast it to other workers
    Both input ids and block table are list of integers
    """
    ret = []
    if is_driver_worker:
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            if seq_group_metadata.block_tables is not None:
                ret.append((seq_data.get_token_ids(), seq_group_metadata.block_tables[seq_id]))
        broadcast_list(is_driver_worker, ret, device)
        return ret
    else:
        return broadcast_list(is_driver_worker, ret, device)
'''

# each model shard needs a lmc instance (vllm_world_size*2)
world_size = 2 * tensor_model_parallel_size * pipeline_model_parallel_size
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=lmc_rank)

def init_comm(
    backend: str,
    lmc_rank: int,
    group_ranks: List[List[int]],
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    distributed_init_method: str,
):
    world_size = 2 * tensor_model_parallel_size * pipeline_model_parallel_size
    torch.distributed.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=lmc_rank)
    
    init_vllm_comm(
        backend, 
        tensor_model_parallel_size, 
        pipeline_model_parallel_size,
        distributed_init_method)
    logger.info("vllm successfully initialized on lmc side")
    
    # initialize four pipes
    self.recv_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    self.recv_signal_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    logger.info("LMCache recv pipe initialized!!!")
    
    self.send_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    self.send_signal_pipe = TorchDistributedPipe(group_ranks, lmc_rank, "gloo")
    logger.info("LMCache send pipe initialized!!!")
    
    return recv_pipe, recv_signal_pipe, send_pipe, send_signal_pipe
    
    
    
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
    for ranks in tp_ranks:
        device_group_TP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")
    
    # Initialize pp ranks
    pp_ranks = []
    num_pp_groups = (vllm_world_size // pipeline_model_parallel_size)
    for i in range(num_pp_groups):
        ranks = list(range(i, vllm_world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    for ranks in pp_ranks:
        device_group_TP = torch.distributed.new_group(ranks, backend=backend)
        cpu_group_TP = torch.distributed.new_group(ranks, backend="gloo")

