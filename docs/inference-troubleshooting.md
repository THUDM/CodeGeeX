# CodeGeeX Inference Troubleshooting

## Common setup issues

| Symptom | Likely cause | Suggested check |
|---|---|---|
| Inference scripts fail before loading the model | The weight path in `configs/codegeex_13b.sh` is still a placeholder | Update the checkpoint path before running the test scripts |
| CUDA OOM on a single GPU run | The full 13B checkpoint exceeds the available VRAM | Use the quantized or multi-GPU path described in the root README |
| Multi-GPU inference does not start | Checkpoints were not converted to the expected model-parallel layout | Run `scripts/convert_ckpt_parallel.sh` before the parallel inference script |
| Docker image works but local Python setup does not | Local dependencies do not match the documented CUDA / PyTorch / DeepSpeed baseline | Re-check the version requirements from the installation section |
| Downloads stop midway | Temporary weight links expired or disk space is insufficient | Refresh the download links and verify you have space for the full archive plus extracted weights |

## Safe debugging order

1. Confirm the checkpoint files are complete.
2. Test one single-GPU or quantized path first.
3. Only then move to model-parallel or Docker-based runs.
