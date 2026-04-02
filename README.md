# NC-KWS-FineTuning

Few-shot keyword spotting fine-tuning for **NC-TCN-20K** (21,689 params).

Add custom wake words (e.g., "marvin", "sheila", "bed") to a pre-trained 12-class KWS model with only 5-20 audio samples per keyword.

## Algorithms

| Method | Key Innovation | Forgetting |
|--------|---------------|------------|
| **Standard LoRA** (baseline) | Rank-4 LoRA on TCN projections | Moderate |
| **NC-ALoRA-PR** (novel) | Adaptive rank + Gradient SNR pruning + Prototype regularization | Low |
| **NC-OPAL** (novel) | Prototype-imprinted head init + Knowledge distillation from frozen teacher | Near-zero |

## Quick Start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrJinHoChoi/NC-KWS-FineTuning/blob/main/notebooks/kws_finetune_comparison.ipynb)

## Structure

```
NC-KWS-FineTuning/
├── nanomamba.py              # NC-TCN-20K model definition
├── core/
│   ├── kws_finetune.py       # Standard LoRA (baseline)
│   ├── kws_finetune_v2.py    # NC-ALoRA-PR (novel)
│   └── kws_finetune_ncopal.py # NC-OPAL (novel)
├── checkpoints/
│   └── best.pt               # Pre-trained NC-TCN-20K (12-class, GSC V2)
└── notebooks/
    └── kws_finetune_comparison.ipynb  # Full comparison notebook
```

## Citation

Part of the NanoMamba / NC-SSM project (Interspeech 2026).
