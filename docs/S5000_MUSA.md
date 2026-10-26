# DreamZero on MTT S5000 / MUSA

This document records the S5000 training and inference-compatibility switches in this branch.

## LIBERO training

`scripts/train/libero_training.sh` requires explicit dataset, output, model, image-encoder, and tokenizer paths. Set `MUSA_VISIBLE_DEVICES` when the runtime needs a restricted card set; otherwise all visible S5000 cards remain available to `torchrun`.

The launcher sets `TF32=false` for MUSA because Transformers rejects `tf32=true` on that backend. Other accelerators retain the upstream `TF32=true` default. Set `TF32` explicitly to override either choice.

## Runtime switches

| Variable | Default | Purpose |
|---|---|---|
| `DREAMZERO_MUSA_LINEAR_BF16_PATCH` | `true` | Split MUSA BF16 `Linear + Bias` into `F.linear` and bias addition. The patch is not installed on CUDA. |
| `DREAMZERO_STATE_SELF_ATTN_MANUAL_SPLIT` | `true` | Use manual state self-attention on MUSA. The switch is ignored on CUDA. |
| `DREAMZERO_CLEAN_FIRST_SDPA_MATH` | `false` | Route clean-first SDPA through the explicit MATH backend when `ATTENTION_BACKEND=torch`. |
| `DREAMZERO_DISABLE_TORCH_COMPILE` | MUSA: `true`; CUDA: `false` | Skip component-level `torch.compile` during inference. |
| `DREAMZERO_GRADIENT_CHECKPOINTING` | `true` | Default activation checkpointing for the action model. |
| `MUSA_FA2_BATCH_CHUNK` | `0` | Optional FA2 batch chunk size; zero keeps the upstream path unchanged. |

These are compatibility controls, not quality-improvement claims. Changes to numerical fallback paths require a finite-value forward/backward check and a matched training run.
