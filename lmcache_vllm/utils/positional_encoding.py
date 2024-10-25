from typing import Dict, Any, Optional, Callable
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from lmcache.logging import init_logger

logger = init_logger(__name__)

def debug_rope(p, q, k):
    return q, k

class BasicReverseRope:
    def __init__(self, rope, rotary_dim, is_neox_style):
        self.rope = rope
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style
        pass

    def do_shuffle(self, t):
        original_shape = t.shape
        t = t.reshape(t.shape[0], -1, self.rotary_dim)

        if self.is_neox_style:
            o1, o2 = torch.chunk(t, 2, dim = -1)
        else:
            o1 = t[..., ::2]
            o2 = t[..., 1::2]

        if self.is_neox_style:
            return torch.cat((o2, o1), dim = -1).reshape(original_shape)
        else:
            return torch.stack((o2, o1), dim = -1).reshape(original_shape)

    def reverse_encode(self, positions, q, k):
        sq = self.do_shuffle(q)
        sk = self.do_shuffle(k)
        nq, nk = self.rope(positions, sq, sk)
        fq = self.do_shuffle(nq)
        fk = self.do_shuffle(nk)
        return fq, fk

    def __call__(self, positions, q, k):
        return self.reverse_encode(positions, q, k)

def validate_rope_params(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        partial_rotary_factor: float = 1.0
    ):
    if rotary_dim != head_size:
        logger.error("Currently KV blending only support rotary_dim == head_size.")
        return False

    if rope_scaling is not None:
        logger.error("Currently KV blending do not support rope scaling.")
        return False

    if dtype is not None:
        logger.error("Currently KV blending do not support special dtype.")
        return False

    if partial_rotary_factor != 1.0:
        logger.error("Currently KV blending do not support rotary factor other than 1.0.")
        return False

    return True

def validate_reverse_correctness(
        rope,
        reverse_rope,
        head_size
    ) -> bool:
    hidden_dim = head_size * 8
    num_tokens = 10

    dumb_q = torch.rand((num_tokens, hidden_dim), device = "cuda", dtype = torch.bfloat16)
    dumb_k = torch.rand((num_tokens, hidden_dim), device = "cuda", dtype = torch.bfloat16)
    positions = torch.arange(num_tokens)

    q1 = dumb_q.clone()
    k1 = dumb_k.clone()
    q1, k1 = rope(positions, q1, k1)
    q1, k1 = reverse_rope(positions, q1, k1)

    max_q_error = (dumb_q - q1).abs().max()
    max_k_error = (dumb_k - k1).abs().max()

    logger.info(f"Max Q error: {max_q_error.item()}")
    logger.info(f"Max K error: {max_k_error.item()}")

    return max_q_error < 0.1 and max_k_error < 0.1


# Main interface
def get_reverse_rope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        partial_rotary_factor: float = 1.0
    ) -> Optional[Callable[..., Any]]:

    # Validate the ROPE parameters
    if not validate_rope_params(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor
        ):
        logger.warning("The rope parameters is not supported! Cannot use cacheblend in this case")
        return None

    rope = vllm_get_rope(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor)

    reverse_rope = BasicReverseRope(rope, rotary_dim, is_neox_style)

    correct = validate_reverse_correctness(rope, reverse_rope, head_size)
    if not correct:
        logger.error("Reverse rotary encoding is not correct! Will disable blending!")
        return None

    return reverse_rope