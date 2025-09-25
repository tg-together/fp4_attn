#!/usr/bin/env python3
import datetime
import pytest
from einops import rearrange
import torch
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from viz_correctness import assert_correctness

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_patch import llama_fp4_attention_forward
from fp4_quant_utils import FP4Quantizer

# Import for pack_scales test
try:
    import b200_attn_fp4
    HAS_B200_MODULE = True
except ImportError:
    HAS_B200_MODULE = False

# Import pack_scales implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from tkfp4_attention_forward import pack_scales_ue4m3_cuda


# Test Configuration - Change these to enable/disable tests
TEST_CONFIG = {
    'gqa': True,                    # Basic GQA test
    'gqa_causal': True,             # GQA with causal masking
    'pack_scales': True,            # Scale packing test
    'gqa_qkfp4': False,             # GQA with QK FP4 quantization
    'gqa_causal_qkfp4': False,      # GQA + Causal + QK FP4
    'gqa_causal_qkpvfp4_fpmask': False,  # Full quantization + FP mask
}

# VIZ_ON_CORRECT = os.getenv("VIZ_ON_CORRECT", "0") == "1"
RESULTS_DIR_PREFIX = os.getenv("FP4_RESULTS_DIR", "/home/austin/results/results/tkfp4")
RESULTS_DIR = f'{RESULTS_DIR_PREFIX}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'


@dataclass
class MockModule:
    """Mock LLaMA attention module with FP4 quantizer setup"""
    layer_idx: int = 1
    num_heads: int = 32
    num_key_value_heads: int = 8
    num_key_value_groups: int = 4
    head_dim: int = 128
    scaling: float = 0.08838834764831845  # 1/sqrt(128)
    attention_dropout: float = 0.0
    training: bool = False
    mean_before_rope: bool = False
    ip: bool = True
    
    def __post_init__(self):
        self.config = MockConfig()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        # Mock projection layers
        hidden_size = self.num_heads * self.head_dim
        kv_hidden_size = self.num_key_value_heads * self.head_dim
        self.q_proj = MockLinear(hidden_size, hidden_size)
        self.k_proj = MockLinear(hidden_size, kv_hidden_size) 
        self.v_proj = MockLinear(hidden_size, kv_hidden_size)
        self.o_proj = MockLinear(hidden_size, hidden_size)


@dataclass
class MockConfig:
    _attn_implementation: str = "eager"
    

class MockLinear:
    def __init__(self, in_features, out_features):
        self.weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=torch.cuda.current_device())
        
    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight)
        

def setup_module(batch_size=2, seq_len=128, num_heads=32, num_kv_heads=8, head_dim=128):
    """Set up mock module with FP4 quantizer"""
    module = MockModule(
        num_heads=num_heads, 
        num_key_value_heads=num_kv_heads,
        num_key_value_groups=num_heads // num_kv_heads,
        head_dim=head_dim
    )
    return module


def generate_inputs(module, batch_size=2, seq_len=128, new_tokens=1):
    """Generate inputs for llama_fp4_attention_forward"""
    device = module.device
    hidden_size = module.num_heads * module.head_dim
    
    # Generate random inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    
    # Position IDs for RoPE
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Generate proper RoPE embeddings following LLaMA implementation
    inv_freq = 1.0 / (10000 ** (torch.arange(0, module.head_dim, 2, device=device).float() / module.head_dim))
    
    # Compute position embeddings
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    position_embeddings = (cos.to(hidden_states.dtype), sin.to(hidden_states.dtype))
    
    # Causal attention mask
    attention_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
    attention_mask = torch.triu(attention_mask, diagonal=1)
    
    return {
        'hidden_states': hidden_states,
        'position_embeddings': position_embeddings, 
        'attention_mask': attention_mask,
        'past_key_value': None,
        'cache_position': None
    }

@pytest.mark.skipif(not TEST_CONFIG['gqa'], reason="GQA test disabled in TEST_CONFIG")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
@pytest.mark.timeout(30)
def test_gqa_attention_correctness(batch_size, seq_len):
    """Isolated GQA test"""
    module = setup_module(batch_size=batch_size, seq_len=seq_len)
    inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
    # Disable quantization and causal masking
    if hasattr(module, 'quantize'):
        delattr(module, 'quantize')
    inputs['attention_mask'] = None
    module.fp_mask = False
    module.training = False
    module.attention_dropout = 0.0
    module.skip_oproj = True
    # input distribution after quantization makes it difficult to test numerical correctness
    module.randn_test = True
    
    # Run both implementations
    module.config._attn_implementation = "eager"
    ref_output, _ = llama_fp4_attention_forward(module, **inputs)
    ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    module.config._attn_implementation = "tkfp4"
    fp4_output, _ = llama_fp4_attention_forward(module, **inputs)
    fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    assert_correctness(
        f"gqa_bs{batch_size}_seq{seq_len}",
        fp4_output, ref_output,
        results_dir=RESULTS_DIR, verbose=True, assert_close=True
    )

@pytest.mark.skipif(not TEST_CONFIG['gqa_causal'], reason="GQA Causal test disabled in TEST_CONFIG")
# @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
# @pytest.mark.parametrize("seq_len", [384, 512, 768, 1024, 1280, 1536, 1792, 2048])
@pytest.mark.timeout(30)
def test_gqa_causal_attention_correctness(batch_size, seq_len):
    """GQA + Causal test"""
    module = setup_module(batch_size=batch_size, seq_len=seq_len)
    inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
    # Disable quantization but keep causal masking
    if hasattr(module, 'quantize'):
        delattr(module, 'quantize')
    module.fp_mask = False
    module.training = False
    module.attention_dropout = 0.0
    module.skip_oproj = True
    # input distribution after quantization makes it difficult to test numerical correctness
    module.randn_test = True
    
    # Run both implementations
    module.config._attn_implementation = "eager"
    # llama_fp4_attention_forward.quantize_enabled = True
    ref_output, _ = llama_fp4_attention_forward(module, **inputs)
    ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    module.config._attn_implementation = "tkfp4"
    # llama_fp4_attention_forward.quantize_enabled = True
    fp4_output, _ = llama_fp4_attention_forward(module, **inputs)
    fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    assert_correctness(
        f"gqa_causal_bs{batch_size}_seq{seq_len}",
        fp4_output, ref_output,
        results_dir=RESULTS_DIR, verbose=True, assert_close=True
    )

# test pack_scales_ue4m3_cuda
@pytest.mark.skipif(not TEST_CONFIG['pack_scales'], reason="pack_scales test disabled in TEST_CONFIG")
@pytest.mark.skipif(not HAS_B200_MODULE, reason="b200_attn_fp4 module not available")
@pytest.mark.parametrize("batch_size,seq_len,num_heads", [
    (1, 128, 32),
    (2, 256, 32),
    (4, 512, 16),
])
def test_pack_scales_correctness(batch_size, seq_len, num_heads):
    """Test that PyTorch pack_scales implementation matches C++ reference"""
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    # Generate random scale tensors
    # Scales are 4D: [batch, heads, seq_len, head_dim // 16] for FP4
    head_dim = 128
    scale_k_dim = head_dim // 16  # 8 for head_dim=128

    # Create 4D test scales in float8_e4m3fn format
    scales_4d = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        scale_k_dim,
        device=device,
        dtype=torch.float32
    ).to(torch.float8_e4m3fn)

    # Test with different configurations
    test_configs = [
        (128, 16, 64),  # Standard config: MN_size=128, block_size=16, MMA_K_TILE=64
        (64, 16, 64),  # Different block size
    ]

    for mn_size, block_size, mma_k_tile in test_configs:
        # Skip configurations that would cause dimension mismatches
        total_rows = batch_size * num_heads * seq_len
        if total_rows % mn_size != 0:
            continue

        # For C++ reference, we need to flatten to 2D
        scales_2d = rearrange(scales_4d, 'b h n d -> (b h n) d')

        # Get C++ reference output (expects 2D)
        cpp_packed_2d = b200_attn_fp4.pack_scales_ue4m3(scales_2d, mn_size, block_size, mma_k_tile)

        # Manually reshape C++ output back to 4D
        k_factor = mn_size // 32
        cpp_packed_4d = rearrange(
            cpp_packed_2d,
            '(b h n) d -> b h n d',
            b=batch_size,
            h=num_heads,
            n=seq_len // k_factor
        )

        # Get PyTorch implementation output (accepts 4D directly)
        py_packed_4d = pack_scales_ue4m3_cuda(scales_4d, mn_size, block_size, mma_k_tile)

        # Compare outputs
        assert cpp_packed_4d.shape == py_packed_4d.shape, \
            f"Shape mismatch: C++ {cpp_packed_4d.shape} vs PyTorch {py_packed_4d.shape}"

        assert torch.allclose(cpp_packed_4d, py_packed_4d), \
            f"pack_scales mismatch for config ({mn_size}, {block_size}, {mma_k_tile})"

    print(f"✓ pack_scales test passed for batch={batch_size}, seq={seq_len}, heads={num_heads}")



@pytest.mark.skipif(not TEST_CONFIG['gqa_qkfp4'], reason="GQA QK FP4 test disabled in TEST_CONFIG")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 256])
@pytest.mark.timeout(30)
def test_gqa_qkfp4_attention_correctness(batch_size, seq_len):
    """GQA + QK FP4 test"""
    module = setup_module(batch_size=batch_size, seq_len=seq_len)
    inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
    # Enable partial FP4 quantization (Q/K only)
    module.quantize = True
    module.fp_mask = False
    inputs['attention_mask'] = None
    # P/V quantization is disabled unless we set quantize_p=True
    
    # Run both implementations
    module.config._attn_implementation = "eager"
    torch.manual_seed(0)
    llama_fp4_attention_forward.quantize_enabled = True
    ref_output, _ = llama_fp4_attention_forward(module, **inputs)
    ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    module.config._attn_implementation = "tkfp4"
    torch.manual_seed(0)
    llama_fp4_attention_forward.quantize_enabled = True
    fp4_output, _ = llama_fp4_attention_forward(module, **inputs)
    fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    assert_correctness(
        f"gqa_qkfp4_bs{batch_size}_seq{seq_len}",
        fp4_output, ref_output,
        results_dir=RESULTS_DIR, verbose=True, assert_close=True
    )

# @pytest.mark.parametrize("batch_size", [1, 2])
# @pytest.mark.parametrize("seq_len", [128, 256])
# @pytest.mark.timeout(30)
# def test_gqa_causal_qkfp4_attention_correctness(batch_size, seq_len):
#     """GQA + Causal + QK FP4 test"""
#     module = setup_module(batch_size=batch_size, seq_len=seq_len)
#     inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
#     # Enable partial FP4 quantization (Q/K only)
#     module.quantize = True
#     module.fp_mask = False
#     # Disable V quantization by setting env vars appropriately
    
#     # Run both implementations
#     module.config._attn_implementation = "eager"
#     ref_output, _ = llama_fp4_attention_forward(module, **inputs)
#     ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
#     module.config._attn_implementation = "tkfp4"
#     fp4_output, _ = llama_fp4_attention_forward(module, **inputs)
#     fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
#     assert_correctness(
#         f"gqa_causal_qkfp4_bs{batch_size}_seq{seq_len}",
#         fp4_output, ref_output,
#         results_dir=RESULTS_DIR, verbose=True, assert_close=True
#     )


# @pytest.mark.parametrize("batch_size", [1, 2])
# # hangs when seqlen < 128
# @pytest.mark.parametrize("seq_len", [128, 256])
# @pytest.mark.timeout(30)
# def test_qga_causal_qkpvfp4_attention_correctness(batch_size, seq_len):
#     """GQA + Causal + QK FP4 + PV FP4 test"""
#     # module.fp_mask = False

#     # Set up module
#     module = setup_module(batch_size=batch_size, seq_len=seq_len)
    
#     # Generate inputs
#     inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
#     # Run PyTorch reference (eager)
#     module.config._attn_implementation = "eager"
#     ref_output, ref_weights = llama_fp4_attention_forward(module, **inputs)
#     ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
#     # Run FP4 attention kernel
#     module.config._attn_implementation = "tkfp4"
#     fp4_output, fp4_weights = llama_fp4_attention_forward(module, **inputs)
#     fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)

#     # Assert correctness with visualization
#     test_name = f"fp4_attn_bs{batch_size}_seq{seq_len}"
#     assert_correctness(
#         name=test_name,
#         out=fp4_output,
#         ref=ref_output,
#         results_dir=RESULTS_DIR,
#         verbose=True,
#         assert_close=True,
#         test_metadata={
#             'batch_size': batch_size,
#             'seq_len': seq_len,
#             'num_heads': module.num_heads,
#             'num_kv_heads': module.num_key_value_heads
#         }
#     )

# @pytest.mark.parametrize("batch_size", [1, 2])
# @pytest.mark.parametrize("seq_len", [128, 256])
# @pytest.mark.timeout(30)
# def test_qga_causal_qkpfp4_fpmask_attention_correctness(batch_size, seq_len):
#     """GQA + Causal + QK FP4 + PV FP4 + FP Mask test"""
#     module = setup_module(batch_size=batch_size, seq_len=seq_len)
#     inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
#     # Enable full quantization + FP mask
#     module.quantize = True
#     module.fp_mask = True
    
#     # Run both implementations
#     module.config._attn_implementation = "eager"
#     ref_output, _ = llama_fp4_attention_forward(module, **inputs)
#     ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
#     module.config._attn_implementation = "tkfp4"
#     fp4_output, _ = llama_fp4_attention_forward(module, **inputs)
#     fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
#     assert_correctness(
#         f"gqa_causal_qkpvfp4_fpmask_bs{batch_size}_seq{seq_len}",
#         fp4_output, ref_output,
#         results_dir=RESULTS_DIR, verbose=True, assert_close=True
#     )


def run_selected_tests():
    """Run only the tests enabled in TEST_CONFIG"""
    import subprocess

    # Build pytest command with markers based on TEST_CONFIG
    enabled_tests = [k for k, v in TEST_CONFIG.items() if v]

    if not enabled_tests:
        print("No tests enabled in TEST_CONFIG")
        return

    print(f"Running enabled tests: {enabled_tests}")
    print("-" * 50)

    # Run pytest with verbose output
    cmd = ["pytest", __file__, "-v", "-s"]

    # Optionally filter to specific test names
    # if 'gqa' in enabled_tests and not 'gqa_causal' in enabled_tests:
    #     cmd.extend(["-k", "test_gqa_attention_correctness"])

    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    parser.add_argument("--run", action="store_true", help="Run selected tests based on TEST_CONFIG")
    args = parser.parse_args()

    if args.run:
        run_selected_tests()
    else:
        # Quick smoke test when run directly
        print("Running FP4 attention smoke test...")
        print(f"Enabled tests in TEST_CONFIG: {[k for k, v in TEST_CONFIG.items() if v]}")
        print("-" * 50)

        module = setup_module(batch_size=1, seq_len=128)
        inputs = generate_inputs(module, batch_size=1, seq_len=128)

        # Test both implementations
        module.config._attn_implementation = "eager"
        ref_out, _ = llama_fp4_attention_forward(module, **inputs)

        module.config._attn_implementation = "tkfp4"
        fp4_out, _ = llama_fp4_attention_forward(module, **inputs)

        print(f"Reference output shape: {ref_out.shape}")
        print(f"FP4 output shape: {fp4_out.shape}")
        print(f"Max diff: {torch.abs(ref_out - fp4_out).max().item():.6f}")
        print("\nSmoke test completed!")
        print("\nTo run the enabled tests, use: python fp4_attention_test.py --run")
        print("Or use pytest directly: pytest fp4_attention_test.py -v")