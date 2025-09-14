# Evaluation Script Fix - Output Path Issue

## Problem
The `eval.py` script was failing with the error:
```
TypeError: simple_evaluate() got an unexpected keyword argument 'output_path'
```

## Root Cause
The `output_path` parameter was being passed to `simple_evaluate()` from lm-evaluation-harness, but this function doesn't accept this parameter. The parameter was being added through:

1. Command line argument parsing (`--output_path`)
2. Sample logging logic that automatically added `output_path` to `eval_kwargs`
3. Unknown argument parsing that could pass `output_path` through

## Solution
1. **Removed `--output_path` argument** from argument parser since it's not supported by `simple_evaluate()`
2. **Filtered out `output_path`** from `eval_kwargs` before passing to `simple_evaluate()`
3. **Simplified sample logging logic** to rely on lm-evaluation-harness internal mechanisms
4. **Fixed JSON serialization** by adding `default=str` to handle non-serializable objects like `dtype`

## Changes Made
- **eval.py:158**: Removed `--output_path` argument parser
- **eval.py:219-220**: Simplified sample logging to only set `log_samples=True`
- **eval.py:139**: Added filtering to remove `output_path` from kwargs
- **eval.py:289**: Added `default=str` to JSON serialization
- **eval.py:293-298**: Updated sample logging documentation

## Usage
### For Individual Sample Logging
```bash
python eval.py --model /path/to/model --device cuda:0 --num_samples 5 --task pile_10k --log_samples --output results.json
```

### For Aggregated Results Only
```bash
python eval.py --model /path/to/model --device cuda:0 --num_samples 5 --task pile_10k --output results.json
```

## Results Location
- **Aggregated results**: Saved to file specified by `--output` parameter
- **Individual samples**: Handled internally by lm-evaluation-harness when `--log_samples` is used

## Verified Working
- ✅ Evaluation runs successfully
- ✅ Results saved to JSON file
- ✅ Sample logging enabled without errors
- ✅ No more `output_path` parameter conflicts