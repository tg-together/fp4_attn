from typing import Optional, Unpack

import torch
from torch import nn

try:
    import b200_attn_fp4
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    print("⚠️  b200_attn_fp4 kernel not available, using stub implementation")

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
    if KERNEL_AVAILABLE:
        attn_out = b200_attn_fp4.attention_forward(
            query,
            key, 
            value,
            query_uq,
            key_uq,
        )
    else:
        # Stub implementation - return zeros with correct shape
        # This will fail correctness tests but enables infrastructure testing
        batch_size, num_heads, seq_len = query_uq.shape[:3]
        head_dim = query_uq.shape[-1]
        attn_out = torch.zeros(
            batch_size, seq_len, num_heads * head_dim,
            device=query_uq.device, 
            dtype=query_uq.dtype
        )

    # no attn_weights returned from kernel
    return attn_out, None