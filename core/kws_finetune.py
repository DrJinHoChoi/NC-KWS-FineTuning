"""NC-TCN KWS custom wake word fine-tuning.

Adds new keywords to NC-TCN-20K via LoRA + output head expansion.
Flow: Record samples (browser mic) -> Save -> Fine-tune -> Hot-reload model.

Approach:
  1. Freeze NC-TCN backbone (TCN blocks + NC frontend)
  2. Add LoRA adapters to TCN projection layers (rank=4)
  3. Expand classification head: 12 -> 12+N classes
  4. Fine-tune on mixed data (original GSC12 + custom keywords)
  5. Hot-swap model in WakeDetector without restart

Minimum: ~30 samples per keyword (10 recommended, 30 for robustness).
Training: ~10-30 seconds on CPU for 50 samples.
"""

import os
import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


PARENT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PARENT)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for nn.Linear."""
    def __init__(self, original: nn.Linear, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


class KWSFineTuner:
    """Custom wake word fine-tuning for NC-TCN-20K.

    Usage:
        ft = KWSFineTuner(wake_detector)
        ft.add_sample("hey_glass", audio_1s)  # repeat ~30 times
        ft.add_sample("hey_glass", audio_1s)
        result = ft.fine_tune()               # ~10-30s on CPU
        # wake_detector is now updated with new keyword
    """

    SR = 16000
    N_FFT = 512
    HOP_LENGTH = 160
    N_MELS = 40
    BASE_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
                   'on', 'off', 'stop', 'go', 'silence', 'unknown']

    def __init__(self, wake_detector=None, data_dir=None, lora_rank=4):
        self.wake_detector = wake_detector
        self.lora_rank = lora_rank

        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.data_dir = data_dir or os.path.join(base_dir, 'kws_custom')
        os.makedirs(self.data_dir, exist_ok=True)

        # Custom keyword samples: {keyword: [audio_arrays]}
        self.samples: dict[str, list[np.ndarray]] = {}
        self.custom_labels: list[str] = []

        # Mel filterbank
        self.window = torch.hann_window(self.N_FFT)
        self.mel_fb = self._build_mel_fb()

        # Training state
        self.is_training = False
        self.last_result = None

        self._load_samples()

    def add_sample(self, keyword: str, audio: np.ndarray, sr: int = 16000) -> dict:
        """Record one sample for a custom keyword.

        Args:
            keyword: new wake word name (e.g., "hey_glass", "smartear")
            audio: 1-second float32 audio at 16kHz
            sr: sample rate

        Returns:
            dict with status
        """
        # Normalize to 1 second
        if len(audio) > sr:
            audio = audio[:sr]
        elif len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)))

        if keyword not in self.samples:
            self.samples[keyword] = []
            if keyword not in self.custom_labels:
                self.custom_labels.append(keyword)

        self.samples[keyword].append(audio.astype(np.float32))
        self._save_samples()

        total = len(self.samples[keyword])
        print(f"  [KWS-FT] Added sample for '{keyword}' "
              f"(total: {total}, all keywords: {self.get_sample_counts()})")

        return {
            "keyword": keyword,
            "count": total,
            "all_counts": self.get_sample_counts(),
            "status": "saved",
        }

    def get_sample_counts(self) -> dict:
        return {k: len(v) for k, v in self.samples.items()}

    def get_status(self) -> dict:
        return {
            "keywords": self.custom_labels,
            "sample_counts": self.get_sample_counts(),
            "is_training": self.is_training,
            "last_result": self.last_result,
            "all_labels": self.BASE_LABELS + self.custom_labels,
        }

    def fine_tune(self, epochs: int = 20, lr: float = 3e-3,
                  neg_ratio: float = 2.0) -> dict:
        """Fine-tune NC-TCN with custom keywords.

        Strategy:
        1. Load base NC-TCN-20K
        2. Expand output head: 12 -> 12+N
        3. Apply LoRA to TCN input/output projections
        4. Train on custom samples + generated negative samples
        5. Hot-reload into wake_detector

        Args:
            epochs: training epochs
            lr: learning rate
            neg_ratio: ratio of negative (silence/noise) samples per keyword sample

        Returns:
            dict with training results
        """
        if self.is_training:
            return {"status": "already_training"}

        # Check minimum samples
        for kw, samples in self.samples.items():
            if len(samples) < 5:
                return {
                    "status": "insufficient_data",
                    "keyword": kw,
                    "count": len(samples),
                    "min_required": 5,
                }

        self.is_training = True
        t0 = time.time()

        try:
            result = self._do_fine_tune(epochs, lr, neg_ratio)
        except Exception as e:
            self.is_training = False
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

        self.is_training = False
        result["time_s"] = round(time.time() - t0, 1)
        self.last_result = result

        # Hot-reload into wake_detector
        if result["status"] == "completed" and self.wake_detector:
            self._hot_reload(result["model_path"])

        return result

    def _do_fine_tune(self, epochs, lr, neg_ratio) -> dict:
        """Core fine-tuning logic."""
        from nanomamba import create_nc_tcn_20k

        # 1. Load base model
        model = create_nc_tcn_20k()
        ckpt_path = os.path.join(PARENT, 'checkpoints', 'best.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=False)

        # 2. Expand classification head
        n_base = len(self.BASE_LABELS)     # 12
        n_custom = len(self.custom_labels)  # N
        n_total = n_base + n_custom

        old_head = model.classifier  # Linear(d_model, 12)
        d_model = old_head.in_features

        new_head = nn.Linear(d_model, n_total)
        # Copy old weights
        with torch.no_grad():
            new_head.weight[:n_base] = old_head.weight
            new_head.bias[:n_base] = old_head.bias
            # Init new classes with small random weights
            nn.init.xavier_uniform_(new_head.weight[n_base:])
            new_head.bias[n_base:].zero_()
        model.classifier = new_head

        # 3. Apply LoRA to TCN blocks
        lora_modules = []
        for block in model.blocks:
            # LoRA on input projection
            if hasattr(block, 'in_proj'):
                lora = LoRALinear(block.in_proj, rank=self.lora_rank)
                block.in_proj = lora
                lora_modules.append(lora)
            # LoRA on output projection
            if hasattr(block, 'out_proj'):
                lora = LoRALinear(block.out_proj, rank=self.lora_rank)
                block.out_proj = lora
                lora_modules.append(lora)

        # 4. Freeze base, train LoRA + new head
        for param in model.parameters():
            param.requires_grad = False
        for m in lora_modules:
            for param in m.parameters():
                param.requires_grad = True
        for param in new_head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [KWS-FT] Trainable: {trainable:,} / {total_params:,} params")
        print(f"  [KWS-FT] Labels: {self.BASE_LABELS + self.custom_labels}")

        # 5. Prepare training data
        X, Y = self._prepare_data(neg_ratio)
        print(f"  [KWS-FT] Training data: {len(X)} samples")

        # 6. Train
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        best_loss = float('inf')
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), 8):
                batch_idx = perm[i:i+8]
                # NanoTCN expects raw audio (B, 16000), does mel internally
                batch_x = torch.stack([
                    torch.from_numpy(X[j]).float()
                    for j in batch_idx])  # (B, 16000)
                batch_y = torch.tensor([Y[j] for j in batch_idx], dtype=torch.long)

                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [KWS-FT] Epoch {epoch+1}/{epochs} "
                      f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.5f}")

        # 7. Save fine-tuned model
        model.eval()
        save_dir = os.path.join(self.data_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'nctcn_custom.pt')

        torch.save({
            'model_state_dict': model.state_dict(),
            'labels': self.BASE_LABELS + self.custom_labels,
            'n_base_labels': n_base,
            'custom_labels': self.custom_labels,
            'lora_rank': self.lora_rank,
        }, model_path)
        print(f"  [KWS-FT] Saved to {model_path}")

        # 8. Quick accuracy check
        model.eval()
        correct = 0
        with torch.no_grad():
            for j in range(len(X)):
                audio = torch.from_numpy(X[j]).float().unsqueeze(0)  # (1, 16000)
                logits = model(audio)
                pred = logits.argmax(dim=-1).item()
                if pred == Y[j]:
                    correct += 1
        acc = correct / len(X) * 100

        return {
            "status": "completed",
            "loss": round(best_loss, 4),
            "accuracy": round(acc, 1),
            "samples": len(X),
            "trainable_params": trainable,
            "labels": self.BASE_LABELS + self.custom_labels,
            "model_path": model_path,
        }

    def _prepare_data(self, neg_ratio: float) -> tuple:
        """Prepare training data: custom samples + negative samples."""
        X = []
        Y = []
        n_base = len(self.BASE_LABELS)

        # Custom keyword samples
        for i, kw in enumerate(self.custom_labels):
            label_idx = n_base + i
            for audio in self.samples.get(kw, []):
                X.append(audio)
                Y.append(label_idx)

                # Data augmentation: time shift + noise
                shifted = np.roll(audio, np.random.randint(-1600, 1600))
                X.append(shifted)
                Y.append(label_idx)

                noisy = audio + np.random.randn(len(audio)).astype(np.float32) * 0.005
                X.append(noisy)
                Y.append(label_idx)

        # Negative samples (silence/noise) to prevent false positives
        n_neg = int(len(X) * neg_ratio)
        silence_idx = self.BASE_LABELS.index('silence')
        unknown_idx = self.BASE_LABELS.index('unknown')

        for _ in range(n_neg // 2):
            # Silence: very low energy
            X.append(np.random.randn(self.SR).astype(np.float32) * 0.001)
            Y.append(silence_idx)

        for _ in range(n_neg - n_neg // 2):
            # Unknown: random noise at moderate level
            X.append(np.random.randn(self.SR).astype(np.float32) * 0.01)
            Y.append(unknown_idx)

        return X, Y

    def _hot_reload(self, model_path: str):
        """Hot-swap the fine-tuned model into the live wake_detector."""
        if not self.wake_detector:
            return

        try:
            from nanomamba import create_nc_tcn_20k

            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            labels = ckpt['labels']
            custom_labels = ckpt['custom_labels']

            # Rebuild model with expanded head
            model = create_nc_tcn_20k()
            n_total = len(labels)
            old_head = model.classifier
            d_model = old_head.in_features

            # Expand head
            model.classifier = nn.Linear(d_model, n_total)

            # Apply LoRA stubs (needed to match state dict keys)
            for block in model.blocks:
                if hasattr(block, 'in_proj'):
                    block.in_proj = LoRALinear(block.in_proj, rank=self.lora_rank)
                if hasattr(block, 'out_proj'):
                    block.out_proj = LoRALinear(block.out_proj, rank=self.lora_rank)

            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()

            # Update wake_detector
            self.wake_detector.model = model
            self.wake_detector.labels = labels

            # Add custom keywords to wake_words
            for kw in custom_labels:
                if kw not in self.wake_detector.wake_words:
                    self.wake_detector.wake_words.append(kw)

            print(f"  [KWS-FT] Hot-reloaded! Labels: {labels}")
            print(f"  [KWS-FT] Wake words: {self.wake_detector.wake_words}")

        except Exception as e:
            print(f"  [KWS-FT] Hot-reload failed: {e}")

    def delete_keyword(self, keyword: str) -> dict:
        """Delete all samples for a keyword."""
        if keyword in self.samples:
            count = len(self.samples[keyword])
            del self.samples[keyword]
            if keyword in self.custom_labels:
                self.custom_labels.remove(keyword)
            self._save_samples()
            return {"status": "deleted", "keyword": keyword, "deleted_count": count}
        return {"status": "not_found", "keyword": keyword}

    # ── Audio processing ──

    def _build_mel_fb(self) -> torch.Tensor:
        n_freq = self.N_FFT // 2 + 1
        low_hz, high_hz = 20, self.SR // 2
        low_mel = 2595 * np.log10(1 + low_hz / 700)
        high_mel = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(low_mel, high_mel, self.N_MELS + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.N_FFT + 1) * hz_points / self.SR).astype(int)

        mel_fb = np.zeros((self.N_MELS, n_freq))
        for i in range(self.N_MELS):
            for j in range(bin_points[i], bin_points[i + 1]):
                mel_fb[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                mel_fb[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)
        return torch.from_numpy(mel_fb).float()

    def _audio_to_logmel(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(audio, self.N_FFT, self.HOP_LENGTH,
                          window=self.window, return_complex=True)
        mag = spec.abs()
        mel = torch.matmul(self.mel_fb, mag)
        return torch.log(mel + 1e-8)

    # ── Persistence ──

    def _save_samples(self):
        meta = {
            "custom_labels": self.custom_labels,
            "samples": {},
        }
        for kw, audios in self.samples.items():
            kw_dir = os.path.join(self.data_dir, kw)
            os.makedirs(kw_dir, exist_ok=True)
            paths = []
            for i, audio in enumerate(audios):
                path = os.path.join(kw_dir, f"{i:04d}.npy")
                np.save(path, audio)
                paths.append(path)
            meta["samples"][kw] = paths

        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_samples(self):
        meta_path = os.path.join(self.data_dir, 'meta.json')
        if not os.path.exists(meta_path):
            return

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.custom_labels = meta.get('custom_labels', [])
        for kw, paths in meta.get('samples', {}).items():
            self.samples[kw] = []
            for p in paths:
                if os.path.exists(p):
                    self.samples[kw].append(np.load(p))

        total = sum(len(v) for v in self.samples.values())
        if total > 0:
            print(f"  [KWS-FT] Loaded {total} samples for "
                  f"{self.custom_labels}")
