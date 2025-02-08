import torch
from typing import Optional, List, Tuple, Dict, Any
from transformers import Qwen2Config

from vllm.attention import AttentionMetadata, Attention
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope 
from vllm.config import CacheConfig

from lmcache_vllm.blend_adapter import do_blend, process_new_request, disable_blend
from lmcache_vllm.utils.positional_encoding import get_reverse_rope

def qwen2_attn_init_with_blend(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[Tuple] = None,
        prefix: str = ""
    ) -> None:
    super(type(self), self).__init__()
    self.hidden_size = hidden_size
    tp_size = get_tensor_model_parallel_world_size()
    self.total_num_heads = num_heads
    assert self.total_num_heads % tp_size == 0
    self.num_heads = self.total_num_heads // tp_size
    self.total_num_kv_heads = num_kv_heads
    if self.total_num_kv_heads >= tp_size:
        # Number of KV heads is greater than TP size, so we partition
        # the KV heads across multiple tensor parallel GPUs.
        assert self.total_num_kv_heads % tp_size == 0
    else:
        # Number of KV heads is less than TP size, so we replicate
        # the KV heads across multiple tensor parallel GPUs.
        assert tp_size % self.total_num_kv_heads == 0
    self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
    self.head_dim = hidden_size // self.total_num_heads
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.rope_theta = rope_theta

    self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config
    )
    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim,
        hidden_size,
        bias=False,
        quant_config=quant_config
    )

    self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
    )
    self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    # Injection for CacheBlend
    self.reverse_rotary_emb = get_reverse_rope(
        self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position,
        base=rope_theta,
        rope_scaling=rope_scaling
    )

    if self.reverse_rotary_emb is None:
        disable_blend()
    # Injection end

def qwen2_attn_forward_with_blend(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # Injection for CacheBlend
    if hasattr(attn_metadata, "blend_metadata"):
        positions = attn_metadata.blend_metadata.positions
    # End of injection

    q, k = self.rotary_emb(positions, q, k)

    # Injection for CacheBlend
    q, k, v, attn_metadata = do_blend(
        q, k, v, attn_metadata,
        self.rotary_emb, self.reverse_rotary_emb
    )
    # End of injection

    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output


def qwen2_decoder_layer_forward_with_blend(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )

    # Injection for CacheBlend
    if hasattr(attn_metadata, "blend_metadata") and attn_metadata.blend_metadata is not None and residual.shape[0] != hidden_states.shape[0]:
        indexes = attn_metadata.blend_metadata.blend_executor.indexes_in_kv
        residual = residual[indexes]
    # End of injection

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual

def qwen2_model_forward_with_blend(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    intermediate_tensors,
    inputs_embeds: Optional[torch.Tensor] = None,
):
    # Injection for CacheBlend
    attn_metadata = process_new_request(input_ids, positions, attn_metadata, kv_caches)
    # End of injection

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            kv_caches[i - self.start_layer],
            attn_metadata,
            residual,
        )
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def inject_qwen2():
    import vllm.model_executor.models.qwen2
    vllm.model_executor.models.qwen2.Qwen2Attention.__init__ = qwen2_attn_init_with_blend
    vllm.model_executor.models.qwen2.Qwen2Attention.forward = qwen2_attn_forward_with_blend
    vllm.model_executor.models.qwen2.Qwen2DecoderLayer.forward = qwen2_decoder_layer_forward_with_blend
    vllm.model_executor.models.qwen2.Qwen2Model.forward = qwen2_model_forward_with_blend