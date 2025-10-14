# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Run the setup script to initialize the conda environment and install dependencies:
```bash
./setup.sh
```

This creates a `fp4` conda environment with Python 3.12 and installs required packages including lm_eval, matplotlib, and glog. It also compiles the CUDA FP4 kernels in the `fast_fp4/` directory.

## Build Commands

### Build CUDA FP4 Extension
```bash
cd fast_fp4/
pip install -e .
```

This compiles the custom CUDA FP4 quantization kernels (`nvfp4sim.cu`, `wrapper.cpp`) using PyTorch's C++ extension framework.

### Test FP4 Kernels
```bash
cd fast_fp4/
python test.py
```

## Core Architecture

This repository implements FP4 quantization for transformer attention mechanisms, specifically targeting LLaMA and Qwen model architectures.

### Key Components

1. **FP4 Quantization Engine** (`fast_fp4/`):
   - `nvfp4sim.cu`: CUDA kernels for FP4 quantization/dequantization with scale search
   - `wrapper.cpp`: Python bindings for CUDA kernels  
   - `setup.py`: PyTorch C++ extension build configuration
   - `test.py`: Kernel testing and validation

2. **Model Patching System**:
   - `llama_patch.py`: Monkey-patches LLaMA attention to use FP4 quantized Q/K/V
   - `qwen3_patch.py`: Monkey-patches Qwen3 attention with FP4 support
   - `fp4_quant_utils.py`: FP4Quantizer class with CPU/GPU fallback implementations

3. **Evaluation Framework**:
   - `eval_ppl.py`: Perplexity evaluation with Hessian recording for quantization sensitivity analysis
   - `lmeval_main.py`: Integration with lm-evaluation-harness for downstream tasks
   - `data_utils.py`: Dataset loading utilities adapted from GPTQ

4. **Benchmarking Scripts**:
   - `ppl_benchmark.sh`: Multi-GPU perplexity benchmarking (baseline, fp4, kvquant modes)
   - `gsm8k_benchmark.sh`: GSM8K mathematical reasoning evaluation  
   - `aime_benchmark.sh`: AIME mathematical competition evaluation
   - `pre_eval_prep.sh`: Hessian recording for quantization analysis

### Quantization Strategy

The codebase implements three quantization approaches:
- **Baseline**: Full precision (no quantization)
- **FP4**: Custom 4-bit floating point quantization with learned scales
- **KVQuant**: Baseline comparison using external KVQuant library

### Model Support

Currently supports:
- LLaMA 3.1 (8B, 70B)
- LLaMA 3.2 (3B-Instruct)  
- LLaMA 3.3 (70B-Instruct)
- Qwen3 (4B, 8B, 4B-Thinking)
- DeepSeek-R1-0528-Qwen3-8B

## Running Benchmarks

### Perplexity Evaluation
```bash
# Run all modes (baseline, fp4, kvquant)
./ppl_benchmark.sh

# Run single mode
MODE="fp4" ./ppl_benchmark.sh
```

### Task-Specific Benchmarks
```bash
./gsm8k_benchmark.sh    # Mathematical reasoning
./aime_benchmark.sh     # Advanced mathematics
```

### Hessian Recording
```bash
./pre_eval_prep.sh      # Record Q/K Hessians for sensitivity analysis
```

## Key Environment Variables

- `HF_HOME="/scratch/huggingface"`: HuggingFace cache directory
- `HF_TOKEN`: HuggingFace API token for model access
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`: GPU memory optimization
- `PYTHONUNBUFFERED=1`: Real-time logging output

## Development Notes

- Benchmarking scripts use multiple GPU configurations (single GPU for smaller models, 2 GPUs for 70B models)
- Logs are saved to `logs/` directory with structured subdirectories by benchmark type
- The codebase includes specific batch size adjustments for different models to handle OOM issues
- Attention patches maintain compatibility with HuggingFace transformers library versions by checking for `past_key_value` argument names