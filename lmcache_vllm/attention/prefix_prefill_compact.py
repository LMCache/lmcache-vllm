# The kernels in this file are adapted from vLLM's prefix_prefill:
# https://github.com/vllm-project/vllm/blob/9ba0817ff1eb514f51cc6de9cb8e16c98d6ee44f/vllm/attention/ops/prefix_prefill.py
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform

if triton.__version__ >= "2.1.0":

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        QK_Out,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_pbs,
        stride_ph,
        stride_pm,
        stride_pn,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        # which of the 8 kv head we are processing
        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_query_len = cur_batch_seq_len - cur_batch_ctx_len

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                       dtype=tl.float32)  # [M,D]
        # acc_p = tl.zeros([BLOCK_M, BLOCK_N],
        #                dtype=q.dtype)  # [M,N]

        # compute query against context (no causal mask here)
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)  # [N]
            # [D,N]
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            # [N,D]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            k_load = tl.load(K_cache + off_k,
                             mask=dim_mask[:, None] &
                             ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                             other=0.0)  # [D,N]

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * k_scale).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                              (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                              -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)  # [M]
            p = tl.exp(qk - m_ij[:, None])  # [M,N]
            l_ij = tl.sum(p, 1)  # [M]
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)  # [M]
            alpha = tl.exp(m_i - m_i_new)  # [M]
            beta = tl.exp(m_ij - m_i_new)  # [M]
            l_i_new = alpha * l_i + beta * l_ij  # [M]

            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # acc_p = acc_p * (acc_scale[:, None]).to(acc_p.dtype)
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)  # [N,D]
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
            else:
                v = v_load
            # p_casted = p.to(acc_p.dtype)
            # acc_p += p_casted
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # p = p.to(acc_p.dtype)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # Lift loop invariants before the loop
        base_off_p = (
            cur_batch * stride_pbs +  # batch index
            cur_head * stride_ph +    # head index
            offs_m[:, None] * stride_pm  # query position
        )

        # compute query against itself (with causal mask)
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] -
                    (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)
            
            off_p = base_off_p + (start_n + offs_n[None, :]) * stride_pn
            # Remove debug print
            # tl.device_print(off_p)
            
            # Add mask for valid positions:
            # 1. Query positions must be within current batch query length
            # 2. Key positions must be within current batch query length
            qk_casted = qk.to(q.dtype)
            tl.store(QK_Out + off_p, 
                    qk_casted,
                    mask=(offs_m[:, None] < cur_batch_query_len) & 
                         ((start_n + offs_n[None, :]) < cur_batch_query_len))

            # qk_casted = qk.to(QK_Out.dtype)
            # qk_casted = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk_casted,
            #               float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # p = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), p,
            #               -1)
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # acc_p = acc_p * acc_scale[:, None]
            # acc_p = acc_p * (acc_scale[:, None]).to(acc_p.dtype)
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
                    # mask=(start_n + offs_n[None, :]) < cur_batch_query_len)
            # tl.store(P + off_p, p_casted)
            # p_casted = p.to(acc_p.dtype)
            # p_masked = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), p_casted, -1.0)
            # acc_p += p_casted
            # p = p.to(acc_p.dtype)
            # acc_p += p
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # Initialize pointers to output and attention pattern
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        # off_p = (
        #     cur_batch * stride_pbs +  # batch index
        #     cur_head * stride_ph +    # head index
        #     offs_m[:, None] * stride_pm +  # query position
        #     offs_m[None, :] * stride_pn    # key position
        # )
        out_ptrs = Out + off_o
        # p_ptrs = P + off_p
        
        # Store output with head_dim mask
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_query_len))
        # Store attention pattern with seq_len mask AND causal mask
        # tl.store(p_ptrs,
        #          acc_p,
        #          mask=(offs_m[:, None] < cur_batch_query_len))
        return

    @torch.inference_mode()
    def context_attention_fwd(q,
                              k,
                              v,
                              p,
                              o,
                              kv_cache_dtype: str,
                              k_cache,
                              v_cache,
                              b_loc,
                              b_start_loc,
                              b_seq_len,
                              b_ctx_len,
                              max_input_len,
                              k_scale: float = 1.0,
                              v_scale: float = 1.0,
                              alibi_slopes=None,
                              sliding_window=None):

        cap = current_platform.get_device_capability()
        BLOCK = 128 if cap[0] >= 8 else 64
        NUM_WARPS = 8

        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        if q.dtype is torch.float32:
            BLOCK = BLOCK // 2

        # Conversion of FP8 Tensor from uint8 storage to
        # appropriate torch.dtype for interpretation by Triton
        if "fp8" in kv_cache_dtype:
            assert (k_cache.dtype == torch.uint8)
            assert (v_cache.dtype == torch.uint8)

            if kv_cache_dtype in ("fp8", "fp8_e4m3"):
                target_dtype = torch.float8_e4m3fn
            elif kv_cache_dtype == "fp8_e5m2":
                target_dtype = torch.float8_e5m2
            else:
                raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        if (k_cache.dtype == torch.uint8
                or v_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
            raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache prefill kernel")

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # round up Lk to a power of 2 - this is required for Triton block size
        Lk_padded = triton.next_power_of_2(Lk)

        sm_scale = 1.0 / (Lq**0.5)
        batch, head = b_seq_len.shape[0], q.shape[1]
        num_queries_per_kv = q.shape[1] // k.shape[1]

        num_blocks = triton.cdiv(max_input_len, BLOCK)
        grid = (batch, head, num_blocks)  # batch, head,

        # 0 means "disable"
        if sliding_window is None or sliding_window <= 0:
            sliding_window = 0
        # print("====================before kernel")
        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}, o: {o.shape}")
        # print(f"k_cache: {k_cache.shape}, v_cache: {v_cache.shape}")
        # print(f"b_loc: {b_loc.shape}, b_start_loc: {b_start_loc.shape}, b_seq_len: {b_seq_len.shape}, b_ctx_len: {b_ctx_len.shape}")
        # print(f"sm_scale: {sm_scale}, k_scale: {k_scale}, v_scale: {v_scale}")
        # print(f"num_queries_per_kv: {num_queries_per_kv}, BLOCK: {BLOCK}, BLOCK_DMODEL: {Lk}, BLOCK_DMODEL_PADDED: {Lk_padded}, BLOCK_N: {BLOCK}, SLIDING_WINDOW: {sliding_window}")
        # print(f"grid: {grid}")

        _fwd_kernel[grid](
            q,
            k,
            v,
            p,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            b_ctx_len,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            p.stride(0),
            p.stride(1),
            p.stride(2),
            p.stride(
                3),  #[batch, num_heads, max_input_len, max_input_len]
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(
                4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SLIDING_WINDOW=sliding_window,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

def forward_prefix_expose(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    context_lens: torch.Tensor,
    max_query_len: int,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Optional[int],
    k_scale: float,
    v_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    # Store raw logits instead of probabilities
    raw_qk = torch.empty(
        (seq_lens_tensor.shape[0], query.shape[1], max_query_len, max_query_len),
        # dtype=torch.float32,  # Always use float32 for logits
        dtype=query.dtype,
        device=value.device
    )

    context_attention_fwd(
        query,
        key,
        value,
        raw_qk,
        output,
        kv_cache_dtype,
        key_cache,
        value_cache,
        block_tables,
        # query_start_loc is (batch_size + 1,)
        query_start_loc[:-1],
        seq_lens_tensor,
        context_lens,
        max_query_len,
        k_scale,
        v_scale,
        alibi_slopes,
        sliding_window,
    )

    # Post-process raw_qk into attention probabilities
    # Apply softmax row-wise
    # attention_probs = torch.nn.functional.softmax(raw_qk, dim=-1)
    
    return output, raw_qk

def convert_logits_to_probs(
    raw_qk: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Convert raw QK logits to proper attention probabilities with masking."""
    batch_size = raw_qk.shape[0]
    num_heads = raw_qk.shape[1]
    
    # Create causal mask
    mask = torch.arange(max_query_len, device=raw_qk.device)[None, :] <= \
           torch.arange(max_query_len, device=raw_qk.device)[:, None]
    
    # Apply masks
    raw_qk = raw_qk.masked_fill(~mask, float('-inf'))
    
    # Apply sequence length mask
    seq_mask = torch.arange(max_query_len, device=raw_qk.device)[None, :] < \
               seq_lens_tensor[:, None]
    raw_qk = raw_qk.masked_fill(~seq_mask[:, None, None, :], float('-inf'))
    
    # Apply softmax
    probs = torch.nn.functional.softmax(raw_qk, dim=-1)
    
    return probs