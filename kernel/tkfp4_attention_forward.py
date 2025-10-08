from typing import Unpack
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

# In your Python script or notebook
import sys
sys.path.insert(0, '/resource/ThunderKittens/kernels/attn/b200_fp4/b200_attn_fp4.cpython-312-x86_64-linux-gnu.so')
import b200_attn_fp4

from einops import rearrange

# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass


TESTING = True

# CUDA STATIC PARAMETERS
TMEM_LANES = 32
BYTES_PER_LANE = 4


class DualQNVFP4AttentionKernelConfig64:
    Q_TILE_HEIGHT = 128
    K_TILE_HEIGHT = 64
    NUM_CONSUMERS = 2
    NUM_CTA = 2

    Q_SCALE_STAGES_PACKED = 4 # <- This is a factor of 2 lower as we have a NUM_CONSUMERS = 2
    K_SCALE_STAGES_PACKED = 16
    Q_SWIZZLE_PER_TILE = 148 * NUM_CONSUMERS # The number of CTA pairs we launch
    Q_SWIZZLE_GROUP_SIZE = NUM_CONSUMERS # The number of consumers within each CTA
    K_SWIZZLE_PERT_TILE = 1
    K_SWIZZLE_GROUP_SIZE = 1

class DualQNVFP4AttentionKernelConfig128:
    Q_TILE_HEIGHT = 128
    K_TILE_HEIGHT = 128
    NUM_CONSUMERS = 2
    NUM_CTA = 2

    Q_SCALE_STAGES_PACKED = 2 # <- This is a factor of 2 lower as we have a NUM_CONSUMERS = 2
    K_SCALE_STAGES_PACKED = 8
    Q_SWIZZLE_PER_TILE = 148 * NUM_CONSUMERS # The number of CTA pairs we launch
    Q_SWIZZLE_GROUP_SIZE = NUM_CONSUMERS # The number of consumers within each CTA
    K_SWIZZLE_PERT_TILE = 1
    K_SWIZZLE_GROUP_SIZE = 1

@torch.no_grad()
def pack_scales_ue4m3_cuda(
    scales_4d: torch.Tensor,
    MN_tile_height: int = 128,
    stages_packed: int = 2,  # The number of groups to pack into a single tile
    tiles_swizzling_factor: int = 2,  # How we should swizzle each packed tile along the sequence dimension
    swizzle_group_size: int  = 2, # The number of micro-tiles to group for swizzling
    tmem_lanes: int = 32,  # The number of rows to store data
    bytes_per_lane: int = 4,  # The number of bytes per column in a TMEM lane
) -> torch.Tensor:
    # Accept 4D tensor [b, h, n, d] and reshape internally
    d = scales_4d.shape[-1]

    PACKED_TILE_WIDTH = d * (MN_tile_height // tmem_lanes) * stages_packed
    if not TESTING:
        assert PACKED_TILE_WIDTH % 32 == 0, "PACKED_TILE_WIDTH must be a multiple of 32 for ThunderKittens"

    INTRA_TILE_WIDTH_SWIZZLE_FACTOR = d // bytes_per_lane
    INTRA_TILE_HEIGHT_SWIZZLE_FACTOR = MN_tile_height // tmem_lanes

    assert (
        INTRA_TILE_WIDTH_SWIZZLE_FACTOR * bytes_per_lane == d
    ), "INTRA_TILE_WIDTH_SWIZZLE_FACTOR * bytes_per_lane must equal d"
    assert (
        INTRA_TILE_HEIGHT_SWIZZLE_FACTOR * tmem_lanes == MN_tile_height
    ), "INTRA_TILE_HEIGHT_SWIZZLE_FACTOR * tmem_lanes must equal MN_tile_height"

    if INTRA_TILE_WIDTH_SWIZZLE_FACTOR > 1:
        # Create each MN_tiles scale data formated so that we follow the correct memory swizzling rules for scales in tmem
        # We swizzle following: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x

        # We also follow scale advancement patterns in: ThunderKittens/include/ops/warp/tensor/mma.cuh
        scales_per_tile = rearrange(
            scales_4d,
            "... (s r tmem_lanes) (p b) -> ... s tmem_lanes (p r b)",
            r=INTRA_TILE_HEIGHT_SWIZZLE_FACTOR,
            b=bytes_per_lane,
            tmem_lanes=tmem_lanes,
        )
    else:
        # Create each MN_tiles scale data formated so that we follow the correct memory swizzling rules for scales in tmem
        # We swizzle following: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x

        scales_per_tile = rearrange(
            scales_4d,
            "... (s r tmem_lanes) d -> ... s tmem_lanes (r d)",
            r=INTRA_TILE_HEIGHT_SWIZZLE_FACTOR,
            tmem_lanes=tmem_lanes,
        )

    # We then interleave the scales for each TILE of TMEM_LANES x d for the number of stages packed
    # The interleaving factor is determined by the swizzling factor

    # For example given 2 SWIZZLING_FACTOR and 4 stages packed, we would have:
    # [0, 2, 4, 6] <- BLOCK 0
    # [1, 3, 5, 7] <- BLOCK 1
    # [8, 10, 12, 14] <- BLOCK 0
    # [9, 11, 13, 15] <- BLOCK 1
    # ...
    # Where each index is a group of 32 x ((MN_tile_height // TMEM_LANES) * d) scales
    # We choose the format so that each CTA load it's own data (This is important for Q) as each CTA load different blocks of q
    # and thus we need to swizzle scales in order to minimize the number of memory accesses.

    # TODO: Make this smarter to reduce memory usage
    # Zero pad so that each CTA is given correct data shapes
    seq_dim = scales_per_tile.shape[-3]
    min_seq_dim = stages_packed * tiles_swizzling_factor * swizzle_group_size
    pad_len = (min_seq_dim - (seq_dim % min_seq_dim))
    if pad_len > 0:
        # Pad the sequence dimension (dim=2)
        pad = (0, 0, 0, 0, 0, pad_len)
        scales_per_tile = F.pad(scales_per_tile, pad=pad, mode='constant', value=0)

    # Now, safely reshape
    interleaved_scales = rearrange(
        scales_per_tile,
        "... (s sp sf gs) t d -> ... (s sf t) (sp gs d)",
        sf=tiles_swizzling_factor,
        sp=stages_packed,
        gs=swizzle_group_size,
    )

    return interleaved_scales


def tkfp4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [b, h, n, d // 2], torch.uint8  2-packed FP4 big endian
    key: torch.Tensor,  # [b, h, n, d // 2], torch.uint8, 2-packed FP4 big endian
    value: torch.Tensor,  # [b, h, n, d], torch.bfloat16
    query_uq: torch.Tensor,  # [b, h, n, d // 2] # unused for now
    key_uq: torch.Tensor,  # [b, h, n, d // 2] # unused for now
    casusal: bool,
    query_scales: torch.Tensor = None,  # torch.float8_e4m3fn # [b, h, n, d // 16]
    query_scales_gs: torch.Tensor = None,  # torch.float32 # [b, h, n]
    key_scales: torch.Tensor = None,  # torch.float8_e4m3fn # [b, h, n, d // 16]
    key_scales_gs: torch.Tensor = None,  # torch.float32 # [b, h, n]
    **kwargs: Unpack[TransformersKwargs],
):

    # update as kernel interface changes
    _query = query.contiguous()
    _key = key.contiguous()
    _value = value.to(torch.bfloat16).contiguous()

    b, h, n, d = _query.shape

    assert 2 * d in [64, 128], "ThundreKittens Dual FP4 kernel supports on d=64,128"
    kernel_config = DualQNVFP4AttentionKernelConfig64() if d == 32 else DualQNVFP4AttentionKernelConfig128()

    o = torch.empty((b, h, n, d * 2), dtype=torch.bfloat16, device=query.device)


    padding = (0, 0, 0, (kernel_config.Q_SWIZZLE_GROUP_SIZE * kernel_config.Q_TILE_HEIGHT) - (n % (kernel_config.Q_SWIZZLE_GROUP_SIZE * kernel_config.Q_TILE_HEIGHT)))
    query_scales = F.pad(query_scales, pad=padding, mode='constant', value=0) # Zero pad so that we have enough
    query_scales = rearrange(query_scales, "b h n d -> (b h n) d") # We need to pack query values


    # The function now accepts 4D tensors directly and returns packed 4D tensors
    packed_query_scales = pack_scales_ue4m3_cuda(query_scales, kernel_config.Q_TILE_HEIGHT, kernel_config.Q_SCALE_STAGES_PACKED, kernel_config.Q_SWIZZLE_PER_TILE, kernel_config.Q_SWIZZLE_GROUP_SIZE, TMEM_LANES, BYTES_PER_LANE)
    packed_key_scales = pack_scales_ue4m3_cuda(key_scales, kernel_config.K_TILE_HEIGHT, kernel_config.K_SCALE_STAGES_PACKED, kernel_config.K_SWIZZLE_PERT_TILE, kernel_config.K_SWIZZLE_GROUP_SIZE, TMEM_LANES, BYTES_PER_LANE)

    _packed_query_scales = packed_query_scales.contiguous()
    _packed_key_scales = packed_key_scales.contiguous()

    _query_scales_gs = query_scales_gs.contiguous()
    _key_scales_gs = key_scales_gs.contiguous()

    scaling_factor = 1 / math.sqrt(64)

    if casusal:
        raise NotImplementedError("Causal mask not supported for now")
        b200_attn_fp4.fwd_attend_ker_64_causal(
            _query,
            _packed_query_scales,
            _key,
            _packed_key_scales,
            _value,
            o,
        )
    else:
        b200_attn_fp4.fwd_attend_ker_64_noncausal(
            _query,
            _packed_query_scales,
            _query_scales_gs,
            _key,
            _packed_key_scales,
            _key_scales_gs,
            _value,
            o,
            float(scaling_factor),
        )

    # no attn_weights returned from kernel
    o = o.transpose(1, 2)
    return o


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
