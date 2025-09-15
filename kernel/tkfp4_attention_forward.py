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
    query: torch.Tensor,                     # torch.int8, 2-packed FP4
    key: torch.Tensor,                       # torch.int8, 2-packed FP4
    value: torch.Tensor,                     # torch.int8, 2-packed FP4
    query_uq: torch.Tensor,                  # torch.float8_e4m3fn
    key_uq: torch.Tensor,                    # torch.float8_e4m3fn
    attention_mask: Optional[torch.Tensor],  # unused
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):

    # update as kernel interface changes
    _query = query.to(torch.bfloat16).contiguous()
    _key = key.to(torch.bfloat16).contiguous()
    _value = value.to(torch.bfloat16).contiguous()
    # _key = repeat_kv(key, module.num_key_value_groups).contiguous()
    # _value = repeat_kv(value, module.num_key_value_groups).contiguous()

    l = torch.empty((_query.shape[0], _query.shape[1], 1, _query.shape[2]), dtype=torch.float32, device=query.device)
    o = torch.empty_like(_query)
    b200_attn_fp4.fwd_attend_ker_128_noncausal(
        _query,
        _key, 
        _value,
        l,
        o,
    )

    o = o.transpose(1, 2).contiguous()

    # no attn_weights returned from kernel
    return o, None