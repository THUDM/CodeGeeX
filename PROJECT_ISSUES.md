# CodeGeeX Project Issues List

This document lists all identified issues in the CodeGeeX project.

## 1. Hardcoded Paths (High Priority) ✅ FIXED

### MindSpore Scripts
**Status: All hardcoded paths have been replaced with configurable options**

All hardcoded paths have been fixed:

- **`codegeex/mindspore/generation_values.py`** ✅
  - Now uses `--output_path` argument (defaults to `./output`)
  - Output file: `output_values.npy`

- **`codegeex/mindspore/generation_humaneval.py`** ✅
  - Now uses `--input_path` for dataset (with smart fallback to repository-relative paths)
  - Now uses `--output_path` for save directory (defaults to `./output`)
  - Language parameter is now properly used (no longer hardcoded to C++)

- **`codegeex/mindspore/generation_finetune.py`** ✅
  - Now uses `--output_path` argument (defaults to `./output`)

- **`codegeex/mindspore/generation_batch.py`** ✅
  - Now uses `--output_path` argument (defaults to `./output`)

- **`codegeex/mindspore/src/dataset.py`** ✅
  - Now uses `eval_data_url` argument (if provided)

- **`codegeex/mindspore/train.py`** ✅
  - Cache paths now use `CODEGEEX_CACHE_BASE` environment variable (defaults to `/home/work/sfs/cache`)
  - `BATCH_JOB_ID` now uses `.get()` with fallback

- **`codegeex/mindspore/scripts/run_modelarts*.py`** ✅
  - Temp directory now uses `MODELARTS_TEMP_DIR` environment variable (defaults to `/home/work/sfs/xx`)
  - Added file existence checks before copying

**New command-line arguments added:**
- `--output_path`: Output directory for generated files (default: `./output`)
- `--input_path`: Input path for data files (optional)

**New environment variables:**
- `CODEGEEX_CACHE_BASE`: Base directory for cache files (default: `/home/work/sfs/cache`)
- `MODELARTS_TEMP_DIR`: Temp directory for ModelArts scripts (default: `/home/work/sfs/xx`)

## 2. Configuration Placeholders (Medium Priority)

Multiple configuration files contain placeholder values that must be set:

- **`configs/codegeex_13b.sh`**: `CHECKPOINT_PATH` placeholder
- **`configs/codegeex_13b_parallel.sh`**: `CHECKPOINT_PATH` placeholder  
- **`configs/codegeex_13b_paddle.sh`**: `CHECKPOINT_PATH` placeholder
- **`scripts/pretrain_codegeex.sh`**: 
  - `HOSTFILE` placeholder
  - `DATA_PATH` placeholder
  - `CKPT_PATH` placeholder
  - `OUTPUT_DIR` placeholder
- **`scripts/finetune_codegeex.sh`**: Same placeholders as pretrain script
- **`codegeex/mindspore/configs/*.sh`**: Multiple config files with `CODE_DATA_DIR` and `<TODO>` placeholders

## 3. TODO Comments / Incomplete Code (Medium Priority)

Multiple TODO comments indicate incomplete work:

- **`codegeex/mindspore/train.py`** (lines 214, 216):
  - TODO: remove after warming-up!
  - TODO: add them back if not for the 1st run!

- **`codegeex/mindspore/src/sat_dataset.py`** (line 81):
  - TODO ARGS comment

- **`codegeex/mindspore/src/dataset.py`** (line 122):
  - TODO: set as current validation set path

- **`codegeex/mindspore/generation_values_1p.py`** (line 166):
  - TODO: add them back if not for the 1st run!

- **`codegeex/mindspore/finetune.py`** (lines 216, 218):
  - TODO: remove after warming-up!
  - TODO: add them back if not for the 1st run!

- **`codegeex/mindspore/generation_1p.py`** (line 166):
  - TODO: add them back if not for the 1st run!

- **`codegeex/mindspore/convertion_1p.py`** (lines 154, 160, 180):
  - Multiple TODOs for checkpoint names and paths

- All generation scripts have TODO comments for setting paths

## 4. Security Issues (High Priority)

- **`codegeex/benchmark/execution.py`** (line 347):
  - Java execution code is commented out with security warning
  - Warning states: "This program exists to execute untrusted model-generated code"
  - Code execution should be sandboxed
  - Currently the `exec_result` is None but code tries to access `.returncode` which will cause AttributeError

- **`codegeex/benchmark/execution.py`** (lines 477-546):
  - `reliability_guard()` function has explicit warning: "This function is NOT a security sandbox"
  - Users should not blindly execute untrusted code

## 5. Known Bugs (Documented)

- **VS Code Extension** (mentioned in README):
  - Bug: If cursor is moved before generation finishes, it may cause issues
  - Location: `vscode-extension/README.md` and `README_zh.md`
  - Status: Acknowledged, team working on making generation faster

## 6. Debug/Test Code Left in Production (Low Priority)

- **`scripts/evaluate_humaneval_x.py`** (lines 47-50):
  - Hardcoded debug values left in code:
    ```python
    #Debugging
    INPUT_FILE='/home/rog0d/Escritorio/CodeGeeX/generations/humaneval_rust_generations.jsonl.gz'
    LANGUAGE='rust'
    ```
  - These override command-line arguments

## 7. Incomplete Implementation (Medium Priority)

- **`tests/test_inference_paddle.py`** (line 149):
  - `raise NotImplementedError("quantize")` - quantization not implemented for Paddle backend

## 8. Path Construction Bug (Low Priority)

- **`scripts/evaluate_humaneval_x.py`** (line 44):
  - Incorrect path join: `os.path.join(MAIN_DIR, "/codegeex/benchmark/humaneval-x/")`
  - Leading slash makes it an absolute path, ignoring MAIN_DIR

## 9. Deprecated/Outdated Information

- **`README.md`** (line 12):
  - Notes that CodeGeeX4 is newer and released
  - Current codebase may be considered legacy version

## 10. Missing Error Handling

- **`codegeex/benchmark/execution.py`** (line 348):
  - Code accesses `exec_result.returncode` when `exec_result` is `None` (line 336)
  - Will cause `AttributeError` - Java execution path is broken

## 11. Hardcoded CUDA Path (Low Priority)

- **`scripts/generate_humaneval_x.sh`** (line 13):
  - `export CUDA_HOME=/usr/local/cuda-11.1/` - hardcoded CUDA version
- **`scripts/translate_humaneval_x.sh`** (line 14):
  - Same hardcoded CUDA path

## 12. Configuration Dependencies

- Scripts require specific environment variables:
  - `BATCH_JOB_ID` (used in train.py and scripts)
  - Various NCCL environment variables
  - Platform-specific paths for Ascend/MindSpore

## Summary by Priority

### Critical (Must Fix Before Production Use)
1. ~~Hardcoded paths in generation scripts~~ ✅ **FIXED**
2. Security issue: Java execution code broken (None.returncode error)
3. Configuration placeholders not set

### High Priority
4. Security warnings for code execution
5. Debug code left in evaluate script
6. Known VS Code extension cursor bug

### Medium Priority
7. Multiple TODO comments indicating incomplete work
8. Hardcoded CUDA paths
9. Path construction bug

### Low Priority
10. Missing quantization implementation for Paddle
11. Deprecated version notice (CodeGeeX4 available)
12. Platform-specific hardcoded paths

