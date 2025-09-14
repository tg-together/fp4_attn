# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements FP4 (4-bit floating point) quantization for attention mechanisms in transformer models, specifically targeting LLaMA architecture. The project includes both CUDA kernels and Python simulation code for FP4 quantization with dual-slice support.

## Build and Setup Commands

### Initial Setup
```bash
bash setup.sh
```
Creates conda environment `fp4`, installs dependencies (lm_eval, matplotlib), configures CUDA 12.9 paths, and builds the custom CUDA extension.

### Build CUDA Extension Only
```bash
cd fast_fp4
pip install -e .
```

### Run Evaluation
```bash
bash run.sh
```
Default run with LLaMA-3-8B on pile_10k task with 100 samples and QKVP quantization.

#### Individual Sample Logging
```bash
python eval.py --model /path/to/model --device cuda:0 --num_samples 5 --task pile_10k --log_samples --output results.json
```
Enables per-sample logging handled internally by lm-evaluation-harness.

### Individual Component Testing
- `bash run_Q.sh` - Test Q quantization only
- `bash run_K.sh` - Test K quantization only  
- `bash run_V.sh` - Test V quantization only
- `bash run_P.sh` - Test P (attention) quantization only

## Key Architecture Components

### CUDA Extension (`fast_fp4/`)
- **nvfp4sim.cu**: Core CUDA kernels for FP4 quantization/dequantization
- **wrapper.cpp**: C++ wrapper exposing kernels to Python
- **setup.py**: PyTorch C++/CUDA extension build configuration
- **test.py**: CUDA kernel testing and benchmarking

### Core Python Modules
- **eval.py**: Main evaluation script using lm_eval framework, integrates model patching
- **llama_patch.py**: Monkey-patches LLaMA attention with FP4 quantization support
- **fp4_quant_utils.py**: FP4Quantizer class with single/dual quantization methods
- **visualize.py**: QKV tensor visualization and analysis tools

### Quantization Strategy
The system implements selective FP4 quantization controlled by the `--quantize` parameter:
- **Q**: Query states with optional dual quantization (high + low precision slices)
- **K**: Key states with zero-point centering options
- **V**: Value states with transpose support
- **P**: Attention weights with optional dual quantization

### Environment Variables
Critical configuration through environment variables:
- `FP4_USE_DUAL_QUANT_Q`: Enable dual Q quantization (default: true)
- `FP4_USE_DUAL_QUANT_ATTN`: Enable dual attention quantization (default: true) 
- `ZERO_POINT`: Zero-point centering for K/V ("min" or "mean")
- `MEAN_BEFORE_ROPE`: Apply mean centering before RoPE (default: false)
- `IP`: Enable incoherence processing (default: true)

### Data Dependencies
The system requires pre-computed statistics files:
- `qk_mean_averages_before_rope.pt`: Layer-wise Q/K mean statistics before RoPE
- `qk_hessians.pt`: Hessian matrices for incoherence processing

### Model Integration
- Patches `transformers.models.llama.modeling_llama.LlamaAttention.forward`
- Maintains compatibility with lm_eval framework
- Supports visualization mode for analysis
- Uses eager attention implementation (not Flash Attention)

## Development Notes

### Testing CUDA Kernels
```bash
cd fast_fp4
python test.py
```

### Debug Mode
The CUDA extension is compiled with debug flags (`-O0 -g -G`) for development.

### Memory Management  
Uses `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` for efficient CUDA memory allocation.

### Visualization
Enable with `--visualize` flag. Generates plots in `plots/` directory for layers 2, 15, 30 showing QKV magnitudes, differences, and MSE.

### Known Issues & Fixes

#### eval.py Output Path Fix
Previous versions had an issue where `--output_path` parameter was passed to `simple_evaluate()` causing:
```
TypeError: simple_evaluate() got an unexpected keyword argument 'output_path'
```

**Fixed in current version by:**
- Removing unsupported `--output_path` parameter
- Using `--output filename.json` for aggregated results
- Relying on lm-evaluation-harness internal sample logging with `--log_samples`

**Current Usage:**
```bash
python eval.py --model /path/to/model --log_samples --output results.json
```