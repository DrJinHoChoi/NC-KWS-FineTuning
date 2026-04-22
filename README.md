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

## Reproducing the Paper

**One-command reviewer script:**

```bash
python reproduce_ncopal.py                 # all 3 methods (lora, alora, opal)
python reproduce_ncopal.py --method opal   # NC-OPAL only
python reproduce_ncopal.py --n-train 30 --n-base 15 --n-test 100
python reproduce_ncopal.py --data-dir /path/to/speech_commands_v0.02
```

Loads the shipped NC-TCN-20K backbone (`checkpoints/best.pt`), fine-tunes
on 3 new keywords (`marvin`, `sheila`, `bed`) with few-shot samples per
method, evaluates base / new / forgetting, and compares to paper numbers.
Exit code `0` within `--tol` (default 3.0 %p); `1` otherwise.

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
