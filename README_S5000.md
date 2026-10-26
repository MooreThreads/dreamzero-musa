# DreamZero S5000 / MUSA Adaptation

This branch adapts the official DreamZero training and inference code for Moore Threads MTT S5000 GPUs using `torch_musa`. It is an engineering compatibility branch, not a new model checkpoint or a claim of upstream benchmark parity.

## Validated environment

| Item | Validated value |
|---|---|
| Accelerator | 8 x MTT S5000, 80 GiB per card |
| Driver | `5.2.0-server` |
| PyTorch | `2.7.1` |
| torch_musa | `2.7.1+84708cc` |
| Training image | `openpi-train:commit-20261008-mudnn-openpi-libero` |
| Training precision | BF16 |
| Distributed backend | MCCL |
| Dataset | LIBERO-LONG LeRobot data (`libero_long_100ep`) |
| Backbone weights | Wan2.2-TI2V-5B |
| Image encoder | Wan2.1-I2V-14B-480P |
| Attention mode | Official attention path with MUSA compatibility fallbacks |

The reviewed revision was re-tested with one S5000 card on 2026-10-21, then with all eight S5000 cards for 1 step and 50 optimizer steps on 2026-10-26. All runs completed model loading, real forward/backward passes, optimizer updates, and checkpoint saves. The original adaptation also completed inference-server startup on S5000.

## Validation status

| Stage | Status | Evidence |
|---|---|---|
| Python imports and model loading | Passed | Real Wan2.2 weights loaded |
| LIBERO-LONG dataset initialization | Passed | `libero_long_100ep`, 1128 dataset steps |
| One-GPU one-step training | Passed | `global_step=1`, finite loss and gradients |
| Eight-GPU 50-step training | Passed on reviewed revision | All 8 ranks completed 50 optimizer steps |
| Inference server startup | Previously passed | Historical server-startup run on S5000 |
| Full training-quality benchmark | Not claimed | No matched evaluation is included in this README |

The one-step re-test recorded:

| Metric | Result |
|---|---:|
| Dynamics loss | `1.521293` |
| Action loss | `0.271813` |
| Train loss | `1.793107` |
| Training step time | `24.238 s` |

The eight-card one-step re-test recorded the same finite losses. Its first-step training time was `38.922 s`; the slowest rank took `40.62 s`. This includes startup and first-step compilation effects and is not a steady-state throughput measurement.

The separate eight-card 50-step run recorded:

| Metric | Result |
|---|---:|
| Optimizer steps | `50` |
| Final loss | `0.9721` |
| Train loss | `1.117573` |
| Total runtime | `416.045 s` |
| Mean time, all 50 steps | `8.161 s/step` |
| Mean time, steps 2-49 | `7.701 s/step` |
| Fastest/slowest steady step | `6.831 s` / `11.990 s` |

## Quickstart

The repository expects model, dataset, tokenizer, and output paths to be supplied explicitly. The LIBERO launcher refuses to start when required paths are missing.

```bash
cd /workspace/openpi

export DS_ACCELERATOR=musa
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export MAX_STEPS=1

export LIBERO_DATA_ROOT=/path/to/libero_long_100ep
export OUTPUT_DIR=/path/to/output
export WAN22_CKPT_DIR=/path/to/Wan2.2-TI2V-5B
export IMAGE_ENCODER_DIR=/path/to/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/path/to/google-umt5-xxl-tokenizer

bash scripts/train/libero_training.sh
```

Start with `NUM_GPUS=1` and `MAX_STEPS=1` before starting a longer run. `TF32` defaults to `false` on MUSA and `true` on other accelerators; it can be overridden explicitly.

Runtime compatibility switches are documented in [`docs/S5000_MUSA.md`](docs/S5000_MUSA.md).

## Scope and limitations

- This branch does not include model weights, datasets, checkpoints, or training outputs.
- The single-step result proves that the MUSA training path executes; it does not establish convergence or task success.
- The MUSA-specific manual state attention, Linear+Bias split, and clean-first SDPA fallback are compatibility paths. They should be evaluated against the corresponding CUDA/reference path before making numerical-parity claims.
- `openpi_client` compatibility behavior matches `openpi-client==0.1.1` for the APIs used by this repository.
- Recreate and pin the runtime image before publishing release artifacts so that system packages cannot drift from `pyproject.toml`.
