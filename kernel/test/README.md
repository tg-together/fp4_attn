# FP4 Attention Testing

Minimal, automated testing infrastructure for FP4 attention kernel validation.

## Quick Start

```bash
make clean && make && cp b200_attn_fp4.cpython-312-x86_64-linux-gnu.so /[your-path]/fp4_attn/
```

```bash
python -m pytest fp4_attention_test.py -v
```

## Architecture

- **`fp4_attention_test.py`**: Core test implementation with pytest parametrization
- **`viz_correctness.py`**: HTML visualization for debugging differences  
- **Environment**: Set `RESULTS_DIR_PREFIX` in `fp4_attention_test.py` to control output location

## Test Flow

1. Mock LLaMA attention module with FP4 quantizer setup
2. Generate inputs for `llama_fp4_attention_forward`
3. Run PyTorch reference (`_attn_implementation="eager"`)
4. Run FP4 kernel (`_attn_implementation="tkfp4"`)
5. Assert correctness with HTML visualization on failure

## Integration

- Drop-in replacement: Set `config._attn_implementation = "tkfp4"` 
- Kernel interface: `tkfp4_attention_forward()` in parent directory
- Falls back to stub implementation when `b200_attn_fp4` unavailable