from typing import Optional, Unpack

import torch
from torch import nn

import b200_attn_fp4

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
def pack_scales_ue4m3_cuda(scales_4d: torch.Tensor, MN_size: int = 128, block_size: int = 16, MMA_K_TILE: int = 64) -> torch.Tensor:
    # Accept 4D tensor [b, h, n, d] and reshape internally
    assert scales_4d.is_cuda and scales_4d.ndim == 4, "Input must be 4D CUDA tensor [b, h, n, d]"
    assert block_size in (16, 32)
    assert MMA_K_TILE in (64, 96)

    # Store original shape for final reshape
    b, h, n, d = scales_4d.shape

    # Flatten to 2D for processing
    scales_u8 = rearrange(scales_4d, 'b h n d -> (b h n) d')

    dev = scales_u8.device
    scale_rows, scale_cols = scales_u8.shape
    assert MN_size > 0 and (scale_rows % MN_size) == 0, "rows must be a multiple of MN_size"

    # Bitcast float8_e4m3fn to uint8 for byte-wise packing
    if scales_u8.dtype == torch.float8_e4m3fn:
        in_bytes = scales_u8.view(torch.uint8).contiguous()
    elif scales_u8.dtype == torch.uint8:
        in_bytes = scales_u8.contiguous()
    else:
        raise RuntimeError("pack_scales_ue4m3_cuda: scales must be torch.uint8 or torch.float8_e4m3fn")

    # Parameters
    k_factor = MN_size // 32
    scale_vec_size = MMA_K_TILE // block_size  # 4 (4X), 2 (2X), or 6 for 96/16, 3 for 96/32 effective per stream

    # Determine pack_size following CPU reference
    if scale_vec_size == 1:
        pack_size = 4
    elif scale_vec_size == 2:
        pack_size = 2
    elif scale_vec_size == 4:
        pack_size = 1
    elif scale_vec_size == 6:
        pack_size = (2 if block_size == 16 else 4)
    else:
        raise RuntimeError("pack_scales_ue4m3_cuda: unsupported scale_vec_size")

    packed_scale_m = 32
    packed_scale_k = (scale_vec_size * pack_size) * k_factor
    col_group = scale_vec_size * pack_size
    # Number of tiles along K (of size scale_vec_size)
    assert (scale_cols % scale_vec_size) == 0, "scale_cols must be divisible by scale_vec_size"
    scale_k_size = scale_cols // scale_vec_size
    # Rows grouped by MN_size
    scale_m_size = scale_rows // MN_size

    # Output shape matches CPU packer
    packed_rows = packed_scale_m * scale_m_size
    packed_cols = ((scale_cols + col_group - 1) // col_group) * packed_scale_k
    out = torch.zeros((packed_rows, packed_cols), dtype=torch.uint8, device=dev)

    # Precompute lane tensors
    m_lanes = torch.arange(MN_size, device=dev, dtype=torch.int32)[:, None]  # [MN_size, 1]
    k_lanes = torch.arange(scale_vec_size, device=dev, dtype=torch.int32)[None, :]  # [1, scale_vec_size]

    for scale_m_tile in range(scale_m_size):
        for scale_k_tile in range(scale_k_size):
            scale_m_offset = scale_m_tile * MN_size
            scale_k_offset = scale_k_tile * scale_vec_size
            packed_scale_m_off = scale_m_tile * packed_scale_m
            packed_scale_k_off = (scale_k_tile // pack_size) * packed_scale_k

            # scale_factor_id = (scale_k_tile % pack_size) * (4 / pack_size)
            scale_factor_id = (scale_k_tile % pack_size) * (4 // pack_size)

            # k_offset selection mirrors CPU reference
            k_offset = -1
            if scale_vec_size == 1:
                k_offset = scale_factor_id  # 0..3
            elif scale_vec_size == 2:
                k_offset = 0 if (scale_factor_id == 0) else 2
            elif scale_vec_size == 4:
                k_offset = 0
            else:  # scale_vec_size == 6
                if block_size == 16:
                    k_offset = 0 if (scale_factor_id == 0) else 6
                else:
                    if scale_factor_id == 0:
                        k_offset = 0
                    elif scale_factor_id == 1:
                        k_offset = 9
                    elif scale_factor_id == 2:
                        k_offset = 6
                    elif scale_factor_id == 3:
                        k_offset = 3
            if k_offset == -1:
                raise RuntimeError("pack_scales_ue4m3_cuda: internal k_offset error")

            # Broadcast lanes to grid [MN_size, scale_vec_size]
            m = m_lanes  # [MN_size, 1]
            k = k_lanes  # [1, scale_vec_size]

            # Compute source indices
            src_m_idx = scale_m_offset + m  # [MN_size, 1]
            src_k_idx = scale_k_offset + k  # [1, scale_vec_size]

            # Compute packed indices
            packed_m_lane = (m % 32)
            k_batch = (k + k_offset) // 4
            k_idx = (k + k_offset) % 4
            packed_k_lane = k_idx + k_batch * (k_factor * 4) + (m // 32) * 4

            packed_m_idx = packed_scale_m_off + packed_m_lane  # [MN_size, 1]
            packed_k_idx = packed_scale_k_off + packed_k_lane  # [MN_size, scale_vec_size]

            # Convert to linear indices for scatter
            packed_m_idx_exp = packed_m_idx.expand(MN_size, scale_vec_size).to(torch.int64)
            packed_k_idx_exp = packed_k_idx.to(torch.int64)
            out_lin_idx = packed_m_idx_exp * packed_cols + packed_k_idx_exp

            src_m_idx_exp = src_m_idx.expand(MN_size, scale_vec_size).to(torch.int64)
            src_k_idx_exp = src_k_idx.expand(MN_size, scale_vec_size).to(torch.int64)
            in_lin_idx = src_m_idx_exp * scale_cols + src_k_idx_exp

            # Scatter copy
            out.view(-1).index_copy_(
                0,
                out_lin_idx.reshape(-1),
                in_bytes.view(-1).index_select(0, in_lin_idx.reshape(-1))
            )

    # Reshape back to 4D with packed dimensions
    # n gets reduced by k_factor, d gets increased by k_factor
    return rearrange(out, '(b h n) d -> b h n d', b=b, h=h, n=n // k_factor, d=d * k_factor)

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

    # Pack scales using PyTorch implementation
    # Note: You may need to adjust MN_size, block_size, MMA_K_TILE based on your kernel requirements
    # The function now accepts 4D tensors directly and returns packed 4D tensors
    packed_query_scales = pack_scales_ue4m3_cuda(_query_scales, 128, 16, 64)  # [b, h, n/4, d*4]
    packed_key_scales = pack_scales_ue4m3_cuda(_key_scales, 128, 16, 64)      # [b, h, n/4, d*4]
    b, qh, n, d = _query_scales.shape
    _, kh, _, _ = _key_scales.shape
    assert packed_query_scales.shape == (b, qh, n//4, d*4)
    assert packed_key_scales.shape == (b, kh, n//4, d*4)

    # only causal mask supported for now
    if attention_mask is not None:
        b200_attn_fp4.fwd_attend_ker_128_causal(
            _query,
            _key, 
            _value,
            packed_query_scales,
            packed_key_scales,
            l,
            o,
        )
    else:
        b200_attn_fp4.fwd_attend_ker_128_noncausal(
            _query,
            _key, 
            _value,
            packed_query_scales,
            packed_key_scales,
            l,
            o,
        )

    o = o.transpose(1, 2).contiguous()

    # no attn_weights returned from kernel
    return o, None