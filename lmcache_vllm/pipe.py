from torch.distributed import Backend
import torch
from typing import List, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


# if the tensor is only one-element and only contains this number
# this means that the sended object is None.
NONE_INT = -150886311
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137

DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
}

INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
}


class BrokenPipeException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TorchDistributedPipe:
    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group

        assert self.device_group is not None
        assert self.rank_in_group <= 1

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.target_rank_for_send = self.ranks[
            (self.rank_in_group + 1) % self.world_size
        ]
        self.target_rank_for_recv = self.ranks[
            (self.rank_in_group - 1) % self.world_size
        ]

        # FIXME: why we need this?
        torch.set_default_device(self.device)

        self.transport_thread = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()

        self.none_tensor = torch.tensor([NONE_INT], device=self.device)

        # On-device tensors to be reused for recv
        self.rcv_metadata_buffer = torch.zeros(
            self.METADATA_LENGTH, dtype=self.METADATA_DTYPE, device=self.device
        )

    def _make_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Create the metadata on based on the input tensor, and move it to GPU.
        The metadata's length is `TorchDistributedPipe.METADATA_LENGTH`.

        Currently, the metadata is a int64 tensor and it includes dtype, number
        of dimensions, and the shape information of the input tensor.


        The information follows the layout below:
        - metadata[0] -- dtype
        - metadata[1] -- number of dimensions
        - metadata[2 : 2+ndims] -- the shape of the input tensor

        Parameters:
            - tensor: the input tensor

        Returns:
            - metadata: the metadata tensor, on self.device
        """
        buffer = torch.empty(self.METADATA_LENGTH, dtype=self.METADATA_DTYPE)
        buffer[0] = DTYPE2INT[tensor.dtype]
        ndims = len(tensor.shape)
        buffer[1] = len(tensor.shape)
        buffer[2 : 2 + ndims] = torch.tensor(
            tensor.shape, dtype=self.METADATA_DTYPE
        )
        return buffer.to(self.device)

    def _prepare_recv_buffer(
        self, d_metadata_buffer: torch.Tensor
    ) -> torch.Tensor:
        """
        Create a buffer to receive the tensor based on the metadata.

        Parameters:
            - d_metadata_buffer: the metadata tensor on self.device

        Returns:
            - buffer: the buffer tensor to receive the tensor, on self.device
        """
        h_buffer = d_metadata_buffer.cpu().numpy()
        dtype = INT2DTYPE[h_buffer[0]]
        ndims = h_buffer[1]
        shape = tuple(h_buffer[2 : 2 + ndims])
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _send_metadata(self, d_metadata_buffer: torch.Tensor):
        """
        Send the metadata buffer to the target rank.
        """
        torch.distributed.send(
            d_metadata_buffer,
            dst=self.target_rank_for_send,
            group=self.device_group,
        )

    def _recv_metadata(self) -> torch.Tensor:
        """
        Receive the metadata buffer from the target rank.

        Returns:
            - metadata_buffer: the metadata buffer tensor, on self.device

        Note:
            The current implementation uses the assumption that there is no
            race conditions during sending/receiving. Therefore, the metadata
            buffer can be reused
        """
        torch.distributed.recv(
            self.rcv_metadata_buffer,
            src=self.target_rank_for_recv,
            group=self.device_group,
        )
        return self.rcv_metadata_buffer

    def _send_impl(self, tensor):
        """
        The actual implementation of sending the tensor to the target rank.
        This function will first send the metadata, and then send the tensor.

        Parameters:
            - tensor: the input tensor to be sent
        """

        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)

        torch.distributed.send(
            tensor, dst=self.target_rank_for_send, group=self.device_group
        )

    def _recv_impl(self) -> torch.Tensor:
        """
        The actual implementation of receiving the tensor from the target rank.
        This function will first receive the metadata, then receive the tensor.

        This function will block if there is no tensor to receive.

        Returns:
            - buffer: the received tensor, on self.device
        """
        d_metadata = self._recv_metadata()
        buffer = self._prepare_recv_buffer(d_metadata)

        torch.distributed.recv(
            buffer, src=self.target_rank_for_recv, group=self.device_group
        )

        return buffer

    def send_tensor_wrapper(self, tensor):
        try:
            """Wrapper for send_tensor_dict"""
            tensor_size = tensor.element_size() * tensor.numel()
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size = self.buffer_size - tensor_size
        except Exception as e:
            logger.error("Encountering exception in KV sending thread")
            logger.error("%s", e)

    def block_if_full(self):
        """
        Block the current thread if the buffer size is larger than 1e9.
        """
        # TODO: replace this 1e9 with a configurable parameter or a constant
        while self.buffer_size > 1e9:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """
        Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is None:
            tensor = self.none_tensor
            tensor_size = 0
        else:
            tensor_size = tensor.element_size() * tensor.numel()

        assert (
            0 < len(tensor.shape) < self.MAX_TENSOR_DIMENSIONS
        ), f"Only support dimensions within 1-{self.MAX_TENSOR_DIMENSIONS}"

        self.block_if_full()

        with self.buffer_size_lock:
            # print("Remaining size:", self.buffer_size)
            self.buffer_size = self.buffer_size + tensor_size

        # prepare the metadata before sending the tensor.
        self.transport_thread.submit(
            self.send_tensor_wrapper,
            tensor,
        )

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread.submit(self._recv_impl)

        try:
            tensor = future.result()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)

        if tensor.numel() == 1 and tensor.item() == NONE_INT:
            return None
        else:
            return tensor

    def close(self):
        """
        Close the pipe and release the resources.
        """
        if (
            hasattr(self, "transport_thread")
            and self.transport_thread is not None
        ):
            self.transport_thread.shutdown()


'''
class Transport:
    def __init__(self, comm_config):
        # TODO(Jiayi): initialize the commuication here    
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
        # TODO(Jiayi): TP/PP/World should be passed in as params
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
    
    def send(self, t: torch.Tensor):
        torch.distributed.send(
            t,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
    
    def recv(self, size: Tuple, dtype: torch.dtype):
        buffer = torch.empty(size, dtype=dtype)
        torch.distributed.recv(
            t,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        return t
    
    
    def send_object(self, obj):
        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")
        # Send object size
        self.send(size_tensor)

        # Send object
        self.send(object_tensor)
        
    def recv_object(self) -> Any:
        # Receive object size
        rank_size = self.recv((1), dtype=torch.long)
        rank_size = rank_size.item()
        
        # Tensor to receive serialized objects into.
        object_tensor = self.recv(rank_size, dtype=torch.uint8)

        obj = pickle.loads(object_tensor.numpy().tobytes())
        return obj
'''