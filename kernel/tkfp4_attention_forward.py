from tkinter import Pack
from typing import Optional, Unpack

import torch
from torch import nn

# import b200_attn_fp4

# remove when gqa is supported
from transformers.models.llama.modeling_llama import repeat_kv

from einops import rearrange

# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass


@torch.no_grad()
def pack_scales_ue4m3_cuda(
    scales_4d: torch.Tensor,
    MN_tile_height: int = 128,
    stages_packed: int = 2,
    num_ctas: int = 2,
    tmem_lanes: int = 32,
    bytes_per_lane: int = 4,
) -> torch.Tensor:
    # Accept 4D tensor [b, h, n, d] and reshape internally
    b, h, n, d = scales_4d.shape
    assert scales_4d.is_cuda and scales_4d.ndim == 4, "Input must be 4D CUDA tensor [b, h, n, d]"

    PACKED_TILE_WIDTH = d * (MN_tile_height // tmem_lanes) * stages_packed
    # assert PACKED_TILE_WIDTH % 32 == 0, "PACKED_TILE_WIDTH must be a multiple of 32 for ThunderKittens"

    TILE_WIDTH_SWIZZLE_FACTOR = d // bytes_per_lane
    TILE_HEIGHT_SWIZZLE_FACTOR = MN_tile_height // tmem_lanes

    assert TILE_WIDTH_SWIZZLE_FACTOR * bytes_per_lane == d, "TILE_WIDTH_SWIZZLE_FACTOR * bytes_per_lane must equal d"
    assert (
        TILE_HEIGHT_SWIZZLE_FACTOR * tmem_lanes == MN_tile_height
    ), "TILE_HEIGHT_SWIZZLE_FACTOR * tmem_lanes must equal MN_tile_height"

    if TILE_WIDTH_SWIZZLE_FACTOR > 1:
        raise NotImplementedError("SWIZZLE_WIDTH > 1 is not supported for now")
    else:
        # Create each MN_tiles scale data formated so that we follow the correct memory swizzling rules for scales in tmem
        # We swizzle following: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x

        scales_per_tile = rearrange(
            scales_4d,
            "b h (s r tmem_lanes) d -> b h s tmem_lanes (r d)",
            r=TILE_HEIGHT_SWIZZLE_FACTOR,
            tmem_lanes=tmem_lanes,
        )

    # We then interleave the scales for each TILE of TMEM_LANES x d for the number of stages packed
    # The interleaving factor is determined by the number of CTA's participating in the kernel

    # For example given 2 CTA's and 4 stages packed, we would have:
    # [0, 2, 4, 6] <- CTA 0
    # [1, 3, 5, 7] <- CTA 1
    # [8, 10, 12, 14] <- CTA 0
    # [9, 11, 13, 15] <- CTA 1
    # ...
    # Where each index is a group of 32 x ((MN_tile_height // TMEM_LANES) * d) scales
    # We choose the format so that each CTA loads it's own data

    interleaved_scales = rearrange(
        scales_per_tile, "b h (s sp cta) t d -> b h s cta t (sp d)", cta=num_ctas, sp=stages_packed
    )

    interleaved_scales = interleaved_scales.reshape(
        b, h, n // (stages_packed * TILE_HEIGHT_SWIZZLE_FACTOR), PACKED_TILE_WIDTH
    )

    return interleaved_scales


def tkfp4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [b, h, n, d // 2], torch.uint8  2-packed FP4 big endian
    key: torch.Tensor,  # [b, h, n, d // 2], torch.uint8, 2-packed FP4 big endian
    value: torch.Tensor,  # [b, h, n, d // 2], torch.uint8, 2-packed FP4 big endian
    query_uq: torch.Tensor,  # [b, h, n, d // 2] # unused for now
    key_uq: torch.Tensor,  # [b, h, n, d // 2] # unused for now
    casusal: bool,
    scaling: float,
    dropout: float = 0.0,
    query_scales: torch.Tensor = None,  # torch.float8_e4m3fn # [b, h, n, d // 16]
    query_scales2: torch.Tensor = None,  # Single float32 value
    key_scales: torch.Tensor = None,  # torch.float8_e4m3fn # [b, h, n, d // 16]
    key_scales2: torch.Tensor = None,  # Single float32 value
    **kwargs: Unpack[TransformersKwargs],
):
    print(f"query: {query.shape}, {query.dtype}")
    print(f"key: {key.shape}, {key.dtype}")
    print(f"value: {value.shape}, {value.dtype}")
    if query_scales is not None:
        print(f"query_scales: {query_scales.shape}, {query_scales.dtype}")
    if key_scales is not None:
        print(f"key_scales: {key_scales.shape}, {key_scales.dtype}")

    # update as kernel interface changes
    _query = query.contiguous()
    _key = key.contiguous()
    _value = value.to(torch.bfloat16).contiguous()

    o = torch.empty(_query.shape, dtype=torch.bfloat16, device=query.device)

    # These are kernel specific meta-parameters
    Q_TILE_HEIGHT = 128
    K_TILE_HEIGHT = 64
    Q_SCALE_STAGES_PACKED = 2
    K_SCALE_STAGES_PACKED = 8
    Q_NUM_CTAS_SPLIT = 1
    K_NUM_CTAS_SPLIT = 2

    # The function now accepts 4D tensors directly and returns packed 4D tensors
    packed_query_scales = pack_scales_ue4m3_cuda(query_scales, Q_TILE_HEIGHT, Q_SCALE_STAGES_PACKED, Q_NUM_CTAS_SPLIT)
    packed_key_scales = pack_scales_ue4m3_cuda(key_scales, K_TILE_HEIGHT, K_SCALE_STAGES_PACKED, K_NUM_CTAS_SPLIT)

    # only causal mask supported for now
    if casusal:
        b200_attn_fp4.fwd_attend_ker_128_causal(
            _query,
            _key,
            _value,
            packed_query_scales,
            packed_key_scales,
            o,
        )
    else:
        b200_attn_fp4.fwd_attend_ker_128_noncausal(
            _query,
            _key,
            _value,
            packed_query_scales,
            packed_key_scales,
            o,
        )

    o = o.transpose(1, 2).contiguous()

    # no attn_weights returned from kernel
    return o, None


if __name__ == "__main__":
    Q_TILE_HEIGHT = 4
    Q_SCALE_STAGES_PACKED = 2
    Q_NUM_CTAS_SPLIT = 2
    tmem_lanes = 2
    bytes_per_lane = 4

    # Create a tensor with a known pattern for easier inspection after packing
    # For example, fill with a ramp along the last dimension
    base_k = torch.arange(4, dtype=torch.float32, device="cuda") / 4  # values: 0, 1/3, 2/3, 1
    base_f_tile_h = torch.arange(32 / Q_TILE_HEIGHT, dtype=torch.float32, device="cuda") * 10
    base_tile_h = torch.arange(Q_TILE_HEIGHT, dtype=torch.float32, device="cuda")

    base_f_tile_h = base_f_tile_h[None, None, :, None].repeat_interleave(Q_TILE_HEIGHT, dim=2)
    base_tile_h = base_tile_h.reshape(1, 1, Q_TILE_HEIGHT, 1).repeat(1, 1, 32 // Q_TILE_HEIGHT, 1)

    scales = base_k.reshape(1, 1, 1, 4).expand(1, 1, 32, 4).contiguous() + base_tile_h + base_f_tile_h

    print(scales)

    packed = pack_scales_ue4m3_cuda(
        scales, Q_TILE_HEIGHT, Q_SCALE_STAGES_PACKED, Q_NUM_CTAS_SPLIT, tmem_lanes, bytes_per_lane
    )

    print(packed)
