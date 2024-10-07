import torch
from typing import Optional, List
from vllm.attention import AttentionMetadata

from lmcache_vllm.blend_adapter import do_blend, process_new_request

def llama_attn_forward_with_blend(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    if hasattr(attn_metadata, "blend_metadata"):
        positions = attn_metadata.blend_metadata.positions

    q, k = self.rotary_emb(positions, q, k)

    q, k, v, attn_metadata = do_blend(q, k, v, attn_metadata)

    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output


def llama_model_forward_with_blend(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    intermediate_tensors,
    inputs_embeds: Optional[torch.Tensor] = None,
):
    attn_metadata = process_new_request(input_ids, positions, attn_metadata)

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
    vllm.model_executor.models.llama.LlamaAttention.forward = llama_attn_forward_with_blend
    vllm.model_executor.models.llama.LlamaModel.forward = llama_model_forward_with_blend

