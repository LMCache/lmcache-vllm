import torch
from typing import Optional, List, Tuple, Dict, Any
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata, Attention
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope 
from vllm.config import CacheConfig

from lmcache_vllm.blend_adapter import do_blend, process_new_request, disable_blend
from lmcache_vllm.utils.positional_encoding import get_reverse_rope

def llama_attn_init_with_blend(
    self,
    config: LlamaConfig,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float = 10000,
    rope_scaling: Optional[Dict[str, Any]] = None,
    max_position_embeddings: int = 8192,
    quant_config: Optional[QuantizationConfig] = None,
    bias: bool = False,
    cache_config: Optional[CacheConfig] = None,
    prefix: str = "",
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
    # MistralConfig has an optional head_dim introduced by Mistral-Nemo
    self.head_dim = getattr(config, "head_dim",
                            self.hidden_size // self.total_num_heads)
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings

    self.qkv_proj = QKVParallelLinear(
        hidden_size=hidden_size,
        head_size=self.head_dim,
        total_num_heads=self.total_num_heads,
        total_num_kv_heads=self.total_num_kv_heads,
        bias=bias,
        quant_config=quant_config,
        prefix=f"{prefix}.qkv_proj",
    )

    self.o_proj = RowParallelLinear(
        input_size=self.total_num_heads * self.head_dim,
        output_size=hidden_size,
        bias=bias,
        quant_config=quant_config,
        prefix=f"{prefix}.o_proj",
    )

    is_neox_style = True
    if quant_config is not None and quant_config.get_name() == "gguf":
        is_neox_style = False

    self.rotary_emb = get_rope(
        self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=is_neox_style,
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
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=is_neox_style,
    )

    if self.reverse_rotary_emb is None:
        disable_blend()
    # Injection end

def llama_attn_forward_with_blend(
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



def llama_decoder_layer_forward_with_blend(
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


    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual

def llama_model_forward_with_blend(
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
            hidden_states = self.get_input_embeddings(input_ids)
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

def inject_llama():
    import vllm.model_executor.models.llama
    vllm.model_executor.models.llama.LlamaAttention.__init__ = llama_attn_init_with_blend
    vllm.model_executor.models.llama.LlamaAttention.forward = llama_attn_forward_with_blend
    vllm.model_executor.models.llama.LlamaDecoderLayer.forward = llama_decoder_layer_forward_with_blend
    vllm.model_executor.models.llama.LlamaModel.forward = llama_model_forward_with_blend

