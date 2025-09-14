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


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.timeout(30)
def test_fp4_attention_correctness(batch_size, seq_len):
    """Test FP4 attention kernel against PyTorch reference"""
    
    # Set up module
    module = setup_module(batch_size=batch_size, seq_len=seq_len)
    
    # Generate inputs
    inputs = generate_inputs(module, batch_size=batch_size, seq_len=seq_len)
    
    # Run PyTorch reference (eager)
    module.config._attn_implementation = "eager"
    ref_output, ref_weights = llama_fp4_attention_forward(module, **inputs)
    ref_output = rearrange(ref_output, 'b n (h d) -> b h n d', h=module.num_heads)
    
    # Run FP4 attention kernel
    module.config._attn_implementation = "tkfp4"
    fp4_output, fp4_weights = llama_fp4_attention_forward(module, **inputs)
    fp4_output = rearrange(fp4_output, 'b n (h d) -> b h n d', h=module.num_heads)

    # Assert correctness with visualization
    test_name = f"fp4_attn_bs{batch_size}_seq{seq_len}"
    assert_correctness(
        name=test_name,
        out=fp4_output,
        ref=ref_output,
        results_dir=RESULTS_DIR,
        verbose=True,
        assert_close=True,
        test_metadata={
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_heads': module.num_heads,
            'num_kv_heads': module.num_key_value_heads
        }
    )


if __name__ == "__main__":
    # Quick smoke test when run directly
    print("Running FP4 attention smoke test...")
    
    module = setup_module(batch_size=1, seq_len=64)
    inputs = generate_inputs(module, batch_size=1, seq_len=64)
    
    # Test both implementations
    module.config._attn_implementation = "eager"
    ref_out, _ = llama_fp4_attention_forward(module, **inputs)
    
    module.config._attn_implementation = "tkfp4" 
    fp4_out, _ = llama_fp4_attention_forward(module, **inputs)
    
    print(f"Reference output shape: {ref_out.shape}")
    print(f"FP4 output shape: {fp4_out.shape}")
    print(f"Max diff: {torch.abs(ref_out - fp4_out).max().item():.6f}")
    print("Smoke test completed!")