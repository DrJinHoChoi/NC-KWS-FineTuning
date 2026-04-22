#!/usr/bin/env python3
"""One-command reproduction of NC-OPAL paper results.

Paper:
  J. H. Choi, "NC-OPAL: Few-Shot Class-Incremental Keyword Spotting on
  Ultra-Tiny Models via Prototype-Imprinted LoRA," 2026 (submitted).

What this script does (for reviewers):
  1. Environment check (torch, torchaudio).
  2. Locates Google Speech Commands v0.02 (or prints download hint).
  3. Loads the pre-trained NC-TCN-20K backbone from
     checkpoints/best.pt (12-class GSC V2).
  4. Runs few-shot fine-tuning for 3 new keywords (marvin, sheila, bed)
     with each method (Standard LoRA / NC-ALoRA-PR / NC-OPAL) and
     reports base acc, new acc, forgetting on a held-out test set.
  5. Compares measured vs paper numbers. Exit 0 on match.

Canonical notebook (Colab, auto-downloads GSC):
    https://colab.research.google.com/github/DrJinHoChoi/NC-KWS-FineTuning/blob/main/notebooks/kws_finetune_comparison.ipynb

Usage:
    python reproduce_ncopal.py                      # all 3 methods, quick
    python reproduce_ncopal.py --method opal        # NC-OPAL only
    python reproduce_ncopal.py --n-train 30         # few-shot budget
    python reproduce_ncopal.py --data-dir /path/to/speech_commands_v0.02

Exit codes: 0 pass, 1 numeric mismatch, 2 deps missing,
            3 dataset missing, 4 checkpoint missing.

License: see LICENSE (academic) / dual-license commercial terms.
Author: Jin Ho Choi (SmartEAR) -- jinhochoi@smartear.co.kr
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------- #
# Paper reference numbers (from paper/ncopal_finetune.tex)              #
# --------------------------------------------------------------------- #
# (base_acc %, new_acc %, forgetting %p) per method
PAPER_TABLE = {
    "lora":   {"base": 85.0, "new": 82.0, "fgt": -6.0},
    "alora":  {"base": 88.0, "new": 85.0, "fgt": -3.0},
    "opal":   {"base": 90.0, "new": 87.0, "fgt": -1.5},
}

NEW_KEYWORDS = ["marvin", "sheila", "bed"]
BASE_LABELS = ["yes", "no", "up", "down", "left", "right",
               "on", "off", "stop", "go", "silence", "unknown"]


def check_env() -> None:
    print("=" * 72)
    print("NC-OPAL -- Few-Shot KWS Fine-Tuning reproduction")
    print("=" * 72)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.platform()}")
    missing = []
    for pkg in ("numpy", "torch", "torchaudio"):
        try:
            m = __import__(pkg)
            print(f"{pkg:<11}: {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(pkg)
            print(f"{pkg:<11}: MISSING")
    if missing:
        print(f"\nMissing: {missing}. "
              "Install: pip install numpy torch torchaudio")
        sys.exit(2)
    import torch
    print(f"CUDA     : {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} GPU)")
    print("=" * 72)


def locate_dataset(data_dir: Path | None) -> Path:
    cands = []
    if data_dir is not None:
        cands.append(data_dir)
    cands += [
        REPO / "data" / "speech_commands_v0.02",
        REPO / "speech_commands_v0.02",
        Path.home() / "speech_commands_v0.02",
    ]
    for c in cands:
        if c.exists() and (c / "yes").is_dir():
            print(f"\n[OK] GSC v0.02 found: {c}")
            return c
    print("\n[ERROR] Google Speech Commands v0.02 not found. Searched:")
    for c in cands:
        print(f"  - {c}")
    print("\nDownload:")
    print("  wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
    print("  mkdir -p data/speech_commands_v0.02")
    print("  tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02/")
    sys.exit(3)


def load_wav_samples(root: Path, keyword: str, n: int, offset: int = 0,
                     sr: int = 16000):
    import numpy as np
    import torchaudio
    kw_dir = root / keyword
    if not kw_dir.is_dir():
        return []
    files = sorted(f for f in kw_dir.iterdir() if f.suffix == ".wav")
    out = []
    for f in files[offset:offset + n]:
        try:
            w, s = torchaudio.load(str(f))
            a = w[0].numpy()
            if len(a) > sr:
                a = a[:sr]
            else:
                a = np.pad(a, (0, sr - len(a)))
            out.append(a.astype("float32"))
        except Exception:
            continue
    return out


def load_backbone(ckpt: Path):
    import torch
    from nanomamba import create_nc_tcn_20k
    if not ckpt.exists():
        print(f"\n[ERROR] Checkpoint not found: {ckpt}")
        print("Expected: checkpoints/best.pt (NC-TCN-20K on 12-class GSC V2)")
        sys.exit(4)
    model = create_nc_tcn_20k()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[OK] NC-TCN-20K loaded: {n:,} params")
    return model


def evaluate_acc(model, samples_per_label, labels, device):
    """Return accuracy over provided samples."""
    import torch
    correct = total = 0
    with torch.no_grad():
        for idx, lbl in enumerate(labels):
            for a in samples_per_label.get(lbl, []):
                x = torch.from_numpy(a).float().unsqueeze(0).to(device)
                pred = model(x).argmax(-1).item()
                correct += int(pred == idx)
                total += 1
    return correct / max(total, 1)


def run_method(method: str, backbone, gsc_root: Path, n_train: int,
               n_base: int, n_test: int, device):
    """Dispatch to the appropriate core/ finetune module and return
    (base_acc, new_acc, forgetting). Methods: 'lora', 'alora', 'opal'.
    """
    # Build few-shot train + eval data
    train_new = {kw: load_wav_samples(gsc_root, kw, n_train)
                 for kw in NEW_KEYWORDS}
    base_words = BASE_LABELS[:10]
    train_base = {kw: load_wav_samples(gsc_root, kw, n_base)
                  for kw in base_words}
    test_all = {}
    for kw in base_words + NEW_KEYWORDS:
        test_all[kw] = load_wav_samples(gsc_root, kw, n_test,
                                        offset=n_train + n_base)

    # Baseline accuracy (backbone only, no fine-tuning)
    backbone.to(device)
    base_pre = evaluate_acc(backbone, test_all, BASE_LABELS, device)

    # Delegate fine-tuning
    if method == "lora":
        from core.kws_finetune import run_finetune as runner
    elif method == "alora":
        from core.kws_finetune_v2 import run_finetune as runner
    elif method == "opal":
        from core.kws_finetune_ncopal import run_finetune as runner
    else:
        raise ValueError(method)

    ft_model = runner(backbone, train_new, train_base, NEW_KEYWORDS,
                      BASE_LABELS, device=device)
    ft_model.to(device)

    all_labels = BASE_LABELS + NEW_KEYWORDS
    base_after = evaluate_acc(ft_model, test_all, BASE_LABELS, device)
    new_acc    = evaluate_acc(ft_model, test_all, NEW_KEYWORDS, device)
    return base_after * 100, new_acc * 100, (base_after - base_pre) * 100


def report(measured: dict, tol: float) -> int:
    print("\n" + "=" * 72)
    print("NC-OPAL reproduction -- Table from ncopal_finetune.tex")
    print("=" * 72)
    print(f"{'Method':<10} {'m_base':>8} {'p_base':>8} "
          f"{'m_new':>8} {'p_new':>8} {'m_fgt':>8} {'p_fgt':>8} "
          f"{'max |d|':>8}")
    print("-" * 72)
    max_dev_all = 0.0
    for method, m in measured.items():
        p = PAPER_TABLE[method]
        devs = [abs(m["base"] - p["base"]),
                abs(m["new"]  - p["new"]),
                abs(m["fgt"]  - p["fgt"])]
        md = max(devs)
        max_dev_all = max(max_dev_all, md)
        flag = "" if md <= tol else " !"
        print(f"{method:<10} "
              f"{m['base']:>8.1f} {p['base']:>8.1f} "
              f"{m['new']:>8.1f} {p['new']:>8.1f} "
              f"{m['fgt']:>+8.1f} {p['fgt']:>+8.1f} "
              f"{md:>+8.2f}{flag}")
    print("=" * 72)
    print(f"Max |delta| vs paper = {max_dev_all:.2f}  (tol = {tol})")
    if max_dev_all <= tol:
        print("[PASS] reproduction within tolerance.")
        return 0
    print("[FAIL] reproduction outside tolerance.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="NC-OPAL reviewer reproduction")
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "checkpoints" / "best.pt")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--method", choices=["lora", "alora", "opal", "all"],
                    default="all")
    ap.add_argument("--n-train", type=int, default=30,
                    help="Few-shot samples per new keyword.")
    ap.add_argument("--n-base", type=int, default=15,
                    help="Base-class anti-forgetting samples per word.")
    ap.add_argument("--n-test", type=int, default=100,
                    help="Test samples per class.")
    ap.add_argument("--tol", type=float, default=3.0,
                    help="Accuracy tolerance (percentage points).")
    args = ap.parse_args()

    check_env()
    t0 = time.time()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gsc = locate_dataset(args.data_dir)
    backbone = load_backbone(args.ckpt)

    methods = (["lora", "alora", "opal"]
               if args.method == "all" else [args.method])
    measured = {}
    for m in methods:
        print(f"\n### Running method: {m} ###")
        import copy
        bb = copy.deepcopy(backbone)
        b, n, f = run_method(m, bb, gsc,
                             args.n_train, args.n_base, args.n_test, device)
        measured[m] = {"base": b, "new": n, "fgt": f}
        print(f"  base={b:.1f}%  new={n:.1f}%  fgt={f:+.1f}%p")

    code = report(measured, tol=args.tol)
    print(f"\nTotal elapsed: {time.time()-t0:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
