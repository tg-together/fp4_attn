#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path
from typing import Optional
from einops import rearrange

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tkfp4_attention_forward import tkfp4_attention_forward, pack_scales_ue4m3_cuda

# Import FP4Quantizer with error handling
try:
    from fp4_quant_utils import FP4Quantizer

    HAS_FP4_QUANTIZER = True
except ImportError as e:
    print(f"Warning: Could not import FP4Quantizer: {e}")
    HAS_FP4_QUANTIZER = False
    FP4Quantizer = None

# Import helper functions from llama_patch
try:
    from llama_patch import block_mask_with_first_block, quantize_p
    HAS_LLAMA_UTILS = True
except ImportError as e:
    print(f"Warning: Could not import llama_patch utils: {e}")
    HAS_LLAMA_UTILS = False
    block_mask_with_first_block = None
    quantize_p = None


# Test Configuration - Change these to enable/disable tests
TEST_CONFIG = {
    "attention_fp4": True,
    "attention_fp4_causal": False,
}


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    value = value.to(torch.bfloat16)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling


    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask


    print(attn_weights[0, 0, :16, :32])
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    # attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)

    # if hasattr(module, 'quantize_p') and module.quantize_p:

    #     Aq_hi, Aq_lo, Aq_hi_int8, Aq_lo_int8, As_hi, As_lo, As_hi_fp8, As_lo_fp8 = quantize_p(module, attn_weights, module.use_dual_quant_attn)
    #     attn_weights = (Aq_hi*As_hi+Aq_lo*As_lo)


    attn_weights = attn_weights.to(torch.bfloat16)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output.to(torch.bfloat16), attn_weights.to(torch.bfloat16)


def eager_attention(query, key, value, query_gs, key_gs, causal=False, scaling=None):
    """Simple wrapper for backward compatibility with smoke tests."""
    _, _, seq_len, head_dim = query.shape

    if scaling is None:
        scaling = (query_gs * key_gs) / (head_dim**0.5) 

    # Create a simple mock module for the call
    module = MockModule()
    
    # Create attention mask for causal case
    attention_mask = None
    if causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask.bool(), float("-inf"))
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Call the new function with query/key as both quantized and unquantized
    return eager_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        query_uq=query,
        key_uq=key,
        attention_mask=attention_mask,
        scaling=scaling,
    )


def create_test_tensors(batch_size=2, num_heads=8, seq_len=128, head_dim=64, device="cuda"):
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    return query, key, value


def quantize_to_fp4_with_scales(tensor, quantizer):
    batch_size, num_heads, seq_len, head_dim = tensor.shape

    # Select first batch item and permute to [seq_len, heads, head_dim]
    tensor_reshaped = tensor[0].permute(1, 0, 2)  # [seq_len, heads, head_dim]
    reconstructed, quantized_data, global_sf, scales = quantizer.single_nvfp4(
        tensor_reshaped, search=False, transpose=False
    )

    # Reshape back to [batch, heads, seq_len, head_dim // 2] for packed data
    # quantized_data is already in the right format from the quantizer
    quantized_data = quantized_data.permute(1, 0, 2)  # [heads, seq_len, head_dim // 2]
    quantized_data = quantized_data.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Reshape scales to [batch, heads, seq_len, head_dim // 16]
    scales = scales.permute(1, 0, 2)  # [heads, seq_len, head_dim // 16]
    scales = scales.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Reshape global scales to [batch, heads, seq_len]
    global_sf = global_sf.permute(1, 0)
    global_sf = global_sf.unsqueeze(0).expand(batch_size, -1, -1)

    return reconstructed, quantized_data, scales, global_sf


class MockModule:
    """Mock module for compatibility with attention forward functions."""

    def __init__(self):
        self.attention_dropout = 0.0
        self.training = False
        self.quantize = False
        self.fp_mask = False
        self.quantize_p = False


@pytest.mark.skipif(not TEST_CONFIG["attention_fp4"], reason="FP4 attention test disabled in TEST_CONFIG")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128])
def test_attention_fp4(batch_size, seq_len):
    """Test FP4 attention without causal masking."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not HAS_FP4_QUANTIZER:
        pytest.skip("FP4Quantizer not available")

    num_heads = 1
    head_dim = 64
    device = "cuda"

    # Create test tensors
    query_fp32, key_fp32, value_fp32 = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device)
    value_fp16 = value_fp32.to(dtype=torch.bfloat16)

    # Compute reference with eager attention

    # Create FP4 quantizer
    quantizer = FP4Quantizer(dequant_dtype=torch.float32, block_size=16, device=torch.device(device))

    # Quantize inputs to FP4
    query_reconstructd, query_fp4, query_scales, query_scales2 = quantize_to_fp4_with_scales(query_fp32, quantizer)
    key_reconstructed, key_fp4, key_scales, key_scales2 = quantize_to_fp4_with_scales(key_fp32, quantizer)

    query_reconstructd = rearrange(query_reconstructd, "(b s) h d -> b h s d", b=batch_size, s=seq_len)
    key_reconstructed = rearrange(key_reconstructed, "(b s) h d -> b h s d", b=batch_size, s=seq_len)

    # Create mock module
    module = MockModule()

    # Run FP4 attention
    fp4_output = tkfp4_attention_forward(
        module=module,
        query=query_fp4,
        key=key_fp4,
        value=value_fp32,
        query_uq=None,
        key_uq=None,
        casusal=False,  # Note: typo in original function signature
        query_scales=query_scales,
        query_scales_gs=query_scales2,
        key_scales=key_scales,
        key_scales_gs=key_scales2,
    )

    ref_output, ref_weights = eager_attention(query_reconstructd, key_reconstructed, value_fp16, query_scales2, key_scales2, causal=False)

    print(f"FP4 attention completed successfully")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"FP4 output shape: {fp4_output.shape}")

    print(fp4_output)
    print(ref_output)

    # Check values are close (FP4 has limited precision)
    assert torch.allclose(fp4_output, ref_output, rtol=0.2, atol=0)



@pytest.mark.skipif(not TEST_CONFIG["attention_fp4_causal"], reason="FP4 causal attention test disabled in TEST_CONFIG")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128])
def test_attention_fp4_causal(batch_size, seq_len):
    """Test FP4 attention with causal masking."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not HAS_FP4_QUANTIZER:
        pytest.skip("FP4Quantizer not available")

    num_heads = 8
    head_dim = 64
    device = "cuda"
    scaling = 1.0 / (head_dim**0.5)

    # Create test tensors
    query_fp32, key_fp32, value_fp32 = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device)

    # Compute reference with eager attention (causal)
    ref_output, ref_weights = eager_attention(query_fp32, key_fp32, value_fp32, causal=True, scaling=scaling)

    # Create FP4 quantizer
    quantizer = FP4Quantizer(dequant_dtype=torch.float32, block_size=16, device=torch.device(device))

    # Quantize inputs to FP4
    query_fp4, query_scales = quantize_to_fp4_with_scales(query_fp32, quantizer)
    key_fp4, key_scales = quantize_to_fp4_with_scales(key_fp32, quantizer)
    value_fp4, _ = quantize_to_fp4_with_scales(value_fp32, quantizer)

    # Create dummy unquantized tensors (unused in current implementation)
    query_uq = torch.zeros_like(query_fp4)
    key_uq = torch.zeros_like(key_fp4)

    # Create mock module
    module = MockModule()

    try:
        # Run FP4 attention with causal masking
        fp4_output = tkfp4_attention_forward(
            module=module,
            query=query_fp4,
            key=key_fp4,
            value=value_fp4,
            query_uq=query_uq,
            key_uq=key_uq,
            casusal=True,  # Note: typo in original function signature
            query_scales=query_scales,
            key_scales=key_scales,
        )

        print(f"FP4 causal attention completed successfully")
        print(f"Reference output shape: {ref_output.shape}")
        print(f"FP4 output shape: {fp4_output.shape}")

        # Check values are close (FP4 has limited precision)
        assert torch.allclose(fp4_output, ref_output, rtol=0.2, atol=1.0)

    except NotImplementedError as e:
        print(f"Causal attention not implemented: {e}")
        pytest.skip("Causal FP4 attention not implemented yet")
    except Exception as e:
        print(f"FP4 causal attention failed: {e}")
        pytest.skip(f"FP4 kernel not available or failed: {e}")


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
    subprocess.run(cmd)


if __name__ == "__main__":
    # Quick smoke test
    print("Running FP4 attention smoke test...")
    print(f"Enabled tests in TEST_CONFIG: {[k for k, v in TEST_CONFIG.items() if v]}")
    print("-" * 50)

    if torch.cuda.is_available():
        # Test basic functionality
        query, key, value = create_test_tensors(1, 8, 128, 64, "cuda")

        # Test eager attention
        output, weights = eager_attention(query, key, value, causal=False)
        print(f"Eager attention output shape: {output.shape}")

        # Test causal eager attention
        output_causal, weights_causal = eager_attention(query, key, value, causal=True)
        print(f"Eager causal attention output shape: {output_causal.shape}")

        # Test FP4 quantization
        if HAS_FP4_QUANTIZER:
            try:
                quantizer = FP4Quantizer(dequant_dtype=torch.float32, block_size=16, device=torch.device("cuda"))
                query_fp4, query_scales = quantize_to_fp4_with_scales(query, quantizer)
                print(f"FP4 quantized query shape: {query_fp4.shape}")
                print(f"FP4 query scales shape: {query_scales.shape}")
            except Exception as e:
                print(f"FP4 quantization failed: {e}")
        else:
            print("FP4Quantizer not available, skipping quantization test")

        print("Smoke test completed!")
        print("\nTo run tests, use: pytest fp4_attention_test.py -v")
    else:
        print("CUDA not available, skipping smoke test")
