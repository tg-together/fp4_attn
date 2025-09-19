from typing import Optional, Unpack

import torch
from torch import nn

import b200_attn_fp4

# remove when gqa is supported
from transformers.models.llama.modeling_llama import repeat_kv

# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass


def tkfp4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,                     # [b, h, n, d // 2], torch.uint8  2-packed FP4 big endian
    key: torch.Tensor,                       # [b, h, n, d // 2], torch.uint8, 2-packed FP4 big endian
    value: torch.Tensor,                     # [b, h, n, d // 2], torch.uint8, 2-packed FP4 big endian
    query_uq: torch.Tensor,                  # unused
    key_uq: torch.Tensor,                    # unused
    attention_mask: Optional[torch.Tensor],  # unused
    scaling: float,
    dropout: float = 0.0,
    query_scales: torch.Tensor = None,       # torch.float8_e4m3fn
    key_scales: torch.Tensor = None,         # torch.float8_e4m3fn
    value_scales: torch.Tensor = None,
    **kwargs: Unpack[TransformersKwargs],
):
    print(f'query: {query.shape}, {query.dtype}')
    print(f'key: {key.shape}, {key.dtype}')
    print(f'value: {value.shape}, {value.dtype}')
    if query_scales is not None:
        print(f'query_scales: {query_scales.shape}, {query_scales.dtype}')
    if key_scales is not None:
        print(f'key_scales: {key_scales.shape}, {key_scales.dtype}')

    # update as kernel interface changes
    _query = query.contiguous()
    _key = key.contiguous()
    _value = value.to(torch.bfloat16).contiguous()
    _query_scales = query_scales.contiguous()
    _key_scales = key_scales.contiguous()
    # _key = repeat_kv(key, module.num_key_value_groups).contiguous()
    # _value = repeat_kv(value, module.num_key_value_groups).contiguous()

    l = torch.empty((_query.shape[0], _query.shape[1], 1, _query.shape[2]), dtype=torch.float32, device=query.device)
    o = torch.empty(_query.shape, dtype=torch.bfloat16, device=query.device)

    # only causal mask supported for now
    if attention_mask is not None:
        b200_attn_fp4.fwd_attend_ker_128_causal(
            _query,
            _key, 
            _value,
            l,
            o,
        )
    else:
        b200_attn_fp4.fwd_attend_ker_128_noncausal(
            _query,
            _key, 
            _value,
            _query_scales,
            _key_scales,
            l,
            o,
        )

    o = o.transpose(1, 2).contiguous()

    # no attn_weights returned from kernel
    return o, None