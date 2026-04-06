#!/usr/bin/env python3
"""
NC-SLU: 4-Way Backbone Comparison on Fluent Speech Commands
Local execution script (CPU/GPU)
Updated to NC-OPAL v2 hyperparameters: r=2, alpha=4, lambda_kd=5, rehearsal=30
"""
import os, sys, time, copy, random, json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Paths (local) ----
REPO = os.path.dirname(os.path.abspath(__file__))
FSC_DIR = os.path.join(os.path.dirname(REPO), 'fluent_speech_commands_dataset', 'fluent_speech_commands_dataset')
sys.path.insert(0, REPO)

from nanomamba import (create_nc_tcn_20k, create_nc_tcn_20k_bi,
                       create_nanomamba_nc_20k, create_nanomamba_nc_20k_bi)
import soundfile as sf

SR = 16000
MAX_LEN = SR * 3  # 3 seconds max
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== NC-OPAL v2 Hyperparameters ==========
LORA_RANK = 2
LORA_ALPHA = 4
LAMBDA_KD = 5.0
KD_TEMP = 3.0
REHEARSAL_PER_CLASS = 30
OLD_AUG_FACTOR = 4
NEW_AUG_FACTOR = 8
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
STAGE1_LR = 4.5e-3
STAGE2_LR = 1.5e-3
BATCH_SIZE = 32
GRAD_CLIP = 1.0
BASE_EPOCHS = 30
# =================================================

print(f"FSC_DIR: {FSC_DIR}")
print(f"Device: {DEVICE}")
print(f"NC-OPAL v2: r={LORA_RANK}, a={LORA_ALPHA}, lambda_KD={LAMBDA_KD}, "
      f"rehearsal={REHEARSAL_PER_CLASS}, Stage1={STAGE1_EPOCHS}ep, Stage2={STAGE2_EPOCHS}ep")

# ---- Verify FSC ----
assert os.path.isdir(FSC_DIR), f"FSC not found at {FSC_DIR}"
assert os.path.isfile(os.path.join(FSC_DIR, 'data', 'train_data.csv')), "train_data.csv not found"

import pandas as pd

# ---- Load FSC data ----
def load_fsc_split(split):
    csv_path = os.path.join(FSC_DIR, 'data', f'{split}_data.csv')
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        intent = f"{row['action']}_{row['object']}_{row['location']}"
        wav_path = os.path.join(FSC_DIR, row['path'])
        data.append({'intent': intent, 'path': wav_path,
                     'action': row['action'], 'object': row['object'],
                     'location': row['location'], 'text': row['transcription']})
    return data

def load_audio(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1: a = a[:, 0]  # mono
    if sr != SR:
        # simple resample via linear interpolation
        ratio = SR / sr
        n_out = int(len(a) * ratio)
        indices = np.linspace(0, len(a)-1, n_out).astype(np.float32)
        idx_f = np.floor(indices).astype(int)
        idx_c = np.minimum(idx_f + 1, len(a)-1)
        a = a[idx_f] * (1 - (indices - idx_f)) + a[idx_c] * (indices - idx_f)
    if len(a) > MAX_LEN: a = a[:MAX_LEN]
    elif len(a) < MAX_LEN: a = np.pad(a, (0, MAX_LEN - len(a)))
    return a.astype(np.float32)

print('Loading FSC metadata...')
train_data = load_fsc_split('train')
val_data = load_fsc_split('valid')
test_data = load_fsc_split('test')

all_intents = sorted(set(d['intent'] for d in train_data))
print(f'Total unique intents: {len(all_intents)}')
for i, intent in enumerate(all_intents):
    count = sum(1 for d in train_data if d['intent'] == intent)
    if i < 5 or i >= len(all_intents) - 3:
        print(f'  [{i:2d}] {intent}: {count} train samples')
    elif i == 5:
        print(f'  ... ({len(all_intents)-8} more)')

# ---- Intent split: 25 base + 6 new ----
random.seed(42)
intents_shuffled = all_intents.copy()
random.shuffle(intents_shuffled)
BASE_INTENTS = intents_shuffled[:25]
NEW_INTENTS = intents_shuffled[25:]
N_SHOT = 20

print(f'\nBase intents: {len(BASE_INTENTS)}')
print(f'New intents ({len(NEW_INTENTS)}): {NEW_INTENTS}')

# ---- Load audio by intent ----
print('\nLoading audio (this may take a few minutes on CPU)...')
def build_dataset(data_list, intents, max_per_intent=None):
    result = defaultdict(list)
    for d in data_list:
        if d['intent'] in intents:
            if max_per_intent and len(result[d['intent']]) >= max_per_intent:
                continue
            try:
                result[d['intent']].append(load_audio(d['path']))
            except Exception as e:
                continue
    return dict(result)

t0 = time.time()
base_train = build_dataset(train_data, BASE_INTENTS)
base_val = build_dataset(val_data, BASE_INTENTS)
base_test = build_dataset(test_data, BASE_INTENTS)
new_all_train = build_dataset(train_data, NEW_INTENTS)
new_train = {k: v[:N_SHOT] for k, v in new_all_train.items()}
new_test = build_dataset(test_data, NEW_INTENTS)

total_base = sum(len(v) for v in base_train.values())
total_new = sum(len(v) for v in new_train.values())
total_test = sum(len(v) for v in base_test.values()) + sum(len(v) for v in new_test.values())
print(f'Loaded in {time.time()-t0:.1f}s')
print(f'  Base train: {total_base} samples across {len(base_train)} intents')
print(f'  New train: {total_new} samples ({N_SHOT}-shot x {len(new_train)} intents)')
print(f'  Test: {total_test} samples')

# ---- Audio Augmentation ----
def aug_time_stretch(a, rate):
    n = int(len(a) / rate)
    indices = np.linspace(0, len(a)-1, n).astype(np.float32)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(a)-1)
    frac = indices - idx_floor
    stretched = a[idx_floor] * (1 - frac) + a[idx_ceil] * frac
    if len(stretched) > MAX_LEN: stretched = stretched[:MAX_LEN]
    elif len(stretched) < MAX_LEN: stretched = np.pad(stretched, (0, MAX_LEN - len(stretched)))
    return stretched.astype(np.float32)

def augment(a):
    out = a.copy()
    if np.random.random() < 0.5:
        out = aug_time_stretch(out, np.random.uniform(0.9, 1.1))
    if np.random.random() < 0.5:
        out = np.roll(out, np.random.randint(-4800, 4800))
    if np.random.random() < 0.5:
        db = np.random.uniform(-4, 4)
        out = out * (10 ** (db / 20))
    if np.random.random() < 0.3:
        ml = int(len(out) * np.random.uniform(0.05, 0.1))
        s = np.random.randint(0, len(out) - ml)
        out[s:s+ml] = 0.0
    out = out + np.random.randn(len(out)).astype(np.float32) * 0.003
    return out.astype(np.float32)

# ---- LoRA (v2: r=2, alpha=4) ----
class LoRALinear(nn.Module):
    def __init__(self, original, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def label_smooth_ce(logits, targets, smoothing=0.1):
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()

# ---- Train base SLU model ----
def train_base_slu(intents, train_data, val_data, epochs=BASE_EPOCHS, model_fn=create_nc_tcn_20k):
    n_cls = len(intents)
    model = model_fn(n_classes=n_cls).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Base SLU model: {n_cls} intents, {n_params:,} params')

    X, Y = [], []
    for i, intent in enumerate(intents):
        for a in train_data.get(intent, []):
            X.append(a); Y.append(i)
    print(f'  Training: {len(X)} samples, {epochs} epochs')

    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X))
        eloss, nb = 0, 0
        for i in range(0, len(X), BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            batch = [augment(X[j]) if np.random.random() < 0.5 else X[j] for j in bi]
            bx = torch.stack([torch.from_numpy(a).float() for a in batch]).to(DEVICE)
            by = torch.tensor([Y[j] for j in bi], dtype=torch.long).to(DEVICE)
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        sched.step()

        if (ep+1) % 10 == 0 or ep == 0:
            model.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for i, intent in enumerate(intents):
                    for a in val_data.get(intent, []):
                        x = torch.from_numpy(a).float().unsqueeze(0).to(DEVICE)
                        pred = model(x).argmax(-1).item()
                        vt += 1; vc += (pred == i)
            print(f'    Epoch {ep+1}/{epochs} loss={eloss/nb:.4f} val_acc={vc/max(vt,1)*100:.1f}%')

    return model

# ---- Evaluate ----
def evaluate_slu(model, intents, test_data):
    model.eval()
    correct, total = 0, 0
    per_intent = {}
    with torch.no_grad():
        for i, intent in enumerate(intents):
            ic, it = 0, 0
            for a in test_data.get(intent, []):
                x = torch.from_numpy(a).float().unsqueeze(0).to(DEVICE)
                pred = model(x).argmax(-1).item()
                it += 1; ic += (pred == i)
            per_intent[intent] = ic / max(it, 1)
            correct += ic; total += it
    return correct / max(total, 1), per_intent

# ---- NC-OPAL v2 for few-shot intent addition ----
def ncopal_slu(base_model, base_intents, new_intents, new_train, base_train_sub):
    """NC-OPAL v2: Proto init + LoRA(r=2) + KD(lambda=5)"""
    model = copy.deepcopy(base_model)
    n_old = len(base_intents)
    n_new = len(new_intents)
    n_total = n_old + n_new
    d = model.classifier.in_features

    # Embedding hook
    emb_hook = {}
    def hook_fn(m, inp, out): emb_hook['e'] = inp[0].detach()
    def get_emb(audio):
        h = model.classifier.register_forward_hook(hook_fn)
        with torch.no_grad(): model(torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE))
        h.remove()
        return emb_hook['e'].squeeze(0)

    # Prototype-initialized head
    old_head = model.classifier
    new_head = nn.Linear(d, n_total).to(DEVICE)
    with torch.no_grad():
        new_head.weight[:n_old] = old_head.weight
        new_head.bias[:n_old] = old_head.bias
        scale = old_head.weight.norm(dim=1).mean().item()
        for i, intent in enumerate(new_intents):
            embs = [get_emb(a) for a in new_train.get(intent, [])]
            if embs:
                proto = F.normalize(torch.stack(embs).mean(0), dim=0)
                new_head.weight[n_old+i] = proto * scale
                new_head.bias[n_old+i] = old_head.bias.mean().item()
    model.classifier = new_head

    # LoRA (v2: r=2, alpha=4)
    loras = []
    for block in model.blocks:
        for attr in ['in_proj', 'out_proj']:
            if hasattr(block, attr):
                orig = getattr(block, attr)
                if isinstance(orig, nn.Linear):
                    lo = LoRALinear(orig, rank=LORA_RANK, alpha=LORA_ALPHA).to(DEVICE)
                    setattr(block, attr, lo); loras.append(lo)

    # Teacher for KD
    teacher = copy.deepcopy(base_model).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # Build training set (v2: rehearsal=30, old_aug=4x, new_aug=8x)
    X, Y = [], []
    for i, intent in enumerate(new_intents):
        li = n_old + i
        for a in new_train.get(intent, []):
            X.append(a); Y.append(li)
            for _ in range(NEW_AUG_FACTOR):
                X.append(augment(a)); Y.append(li)
    for i, intent in enumerate(base_intents):
        for a in base_train_sub.get(intent, [])[:REHEARSAL_PER_CLASS]:
            X.append(a); Y.append(i)
            for _ in range(OLD_AUG_FACTOR):
                X.append(augment(a)); Y.append(i)

    n_new_s = sum(1 for y in Y if y >= n_old)
    n_old_s = sum(1 for y in Y if y < n_old)
    print(f'  Train: {len(X)} samples (new={n_new_s}, old={n_old_s})')

    # Stage 1: Head only
    for p in model.parameters(): p.requires_grad = False
    for p in new_head.parameters(): p.requires_grad = True
    opt1 = torch.optim.AdamW(new_head.parameters(), lr=STAGE1_LR, weight_decay=0.01)
    model.train()
    for ep in range(STAGE1_EPOCHS):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            bx = torch.stack([torch.from_numpy(X[j]).float() for j in bi]).to(DEVICE)
            by = torch.tensor([Y[j] for j in bi], dtype=torch.long).to(DEVICE)
            loss = label_smooth_ce(model(bx), by, 0.1)
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(new_head.parameters(), GRAD_CLIP)
            opt1.step()
    print(f'  Stage 1 done (head-only, {STAGE1_EPOCHS} epochs)')

    # Stage 2: LoRA + Head + KD (v2: lambda_kd=5, 20 epochs)
    for m in loras:
        for p in m.parameters(): p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {'params': [p for m in loras for p in m.parameters()], 'lr': STAGE2_LR},
        {'params': new_head.parameters(), 'lr': STAGE2_LR * 0.5},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)

    for ep in range(STAGE2_EPOCHS):
        perm = np.random.permutation(len(X))
        ecl, ekd, nbt = 0, 0, 0
        for i in range(0, len(X), BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            bx = torch.stack([torch.from_numpy(X[j]).float() for j in bi]).to(DEVICE)
            by = torch.tensor([Y[j] for j in bi], dtype=torch.long).to(DEVICE)
            logits = model(bx)
            cls_loss = label_smooth_ce(logits, by, 0.05)

            kd_loss = torch.tensor(0.0).to(DEVICE)
            old_mask = by < n_old
            if old_mask.any():
                with torch.no_grad(): t_logits = teacher(bx[old_mask])
                s_base = logits[old_mask][:, :n_old] / KD_TEMP
                t_base = t_logits / KD_TEMP
                kd_loss = F.kl_div(F.log_softmax(s_base, -1),
                                   F.softmax(t_base, -1),
                                   reduction='batchmean') * (KD_TEMP**2)

            kd_w = LAMBDA_KD * min(1.0, ep / 5.0)
            total = cls_loss + kd_w * kd_loss
            opt2.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt2.step()
            ecl += cls_loss.item(); ekd += kd_loss.item(); nbt += 1
        sched.step()
        if (ep+1) % 5 == 0:
            print(f'    Stage2 Epoch {ep+1}/{STAGE2_EPOCHS} cls={ecl/nbt:.4f} kd={ekd/nbt:.4f}')

    return model, base_intents + new_intents

# ---- Finetune baseline ----
def finetune_slu(base_model, base_intents, new_intents, new_train, base_train_sub):
    model = copy.deepcopy(base_model)
    n_old = len(base_intents)
    n_total = n_old + len(new_intents)
    d = model.classifier.in_features

    old_head = model.classifier
    new_head = nn.Linear(d, n_total).to(DEVICE)
    with torch.no_grad():
        new_head.weight[:n_old] = old_head.weight
        new_head.bias[:n_old] = old_head.bias
        nn.init.xavier_uniform_(new_head.weight[n_old:])
        new_head.bias[n_old:].zero_()
    model.classifier = new_head

    X, Y = [], []
    for i, intent in enumerate(new_intents):
        li = n_old + i
        for a in new_train.get(intent, []):
            X.append(a); Y.append(li)
            for _ in range(NEW_AUG_FACTOR):
                X.append(augment(a)); Y.append(li)
    for i, intent in enumerate(base_intents):
        for a in base_train_sub.get(intent, [])[:REHEARSAL_PER_CLASS]:
            X.append(a); Y.append(i)
            for _ in range(OLD_AUG_FACTOR):
                X.append(augment(a)); Y.append(i)

    epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        eloss, nb = 0, 0
        for i in range(0, len(X), BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            bx = torch.stack([torch.from_numpy(X[j]).float() for j in bi]).to(DEVICE)
            by = torch.tensor([Y[j] for j in bi], dtype=torch.long).to(DEVICE)
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            eloss += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 10 == 0:
            print(f'    Finetune Epoch {ep+1}/{epochs} loss={eloss/nb:.4f}')
    return model, base_intents + new_intents


# ====================
# MAIN EXPERIMENT
# ====================
print('\n' + '='*70)
print('  NC-SLU: 4-Way Backbone Comparison (NC-OPAL v2)')
print(f'  Base: {len(BASE_INTENTS)} intents | New: {len(NEW_INTENTS)} intents ({N_SHOT}-shot)')
print(f'  Device: {DEVICE}')
print(f'  NC-OPAL v2: r={LORA_RANK}, lambda_KD={LAMBDA_KD}, rehearsal={REHEARSAL_PER_CLASS}')
print('='*70)

all_test_data = {**base_test, **new_test}
base_train_sub = {k: v[:REHEARSAL_PER_CLASS] for k, v in base_train.items()}
results = {}
base_accs = {}

BACKBONES = [
    ('NC-TCN',     create_nc_tcn_20k,        'Causal TCN, RF=15'),
    ('NC-TCN-Bi',  create_nc_tcn_20k_bi,     'Bidirectional TCN'),
    ('NC-SSM',     create_nanomamba_nc_20k,   'Selective SSM, RF=inf'),
    ('NC-SSM-Bi',  create_nanomamba_nc_20k_bi,'Bidirectional SSM'),
]

exp_start = time.time()

for bb_idx, (bb_name, bb_fn, bb_desc) in enumerate(BACKBONES):
    print(f'\n{"="*60}')
    print(f'  [{bb_idx+1}/4] BACKBONE: {bb_name} -{bb_desc}')
    print(f'{"="*60}')

    # Phase 1: Train base model
    print(f'\n[Phase 1] Training base model ({bb_name})...')
    t_start = time.time()
    base_model = train_base_slu(BASE_INTENTS, base_train, base_val, epochs=BASE_EPOCHS, model_fn=bb_fn)
    base_acc, base_per = evaluate_slu(base_model, BASE_INTENTS, base_test)
    t_base = time.time() - t_start
    base_accs[bb_name] = base_acc
    print(f'  [OK] Base accuracy: {base_acc*100:.1f}% [{t_base:.0f}s]')

    # Phase 2a: Finetune baseline (first backbone only)
    if bb_idx == 0:
        print(f'\n[Phase 2a] Finetune baseline ({bb_name})...')
        t_start = time.time()
        ft_model, ft_intents = finetune_slu(base_model, BASE_INTENTS, NEW_INTENTS, new_train, base_train_sub)
        ft_acc, ft_per = evaluate_slu(ft_model, ft_intents, all_test_data)
        ft_base_acc = np.mean([ft_per[i] for i in BASE_INTENTS])
        ft_new_acc = np.mean([ft_per[i] for i in NEW_INTENTS])
        t_ft = time.time() - t_start

        n_params = sum(p.numel() for p in bb_fn(n_classes=31).parameters())
        results[f'Finetune ({bb_name})'] = {
            'overall': ft_acc, 'base': ft_base_acc, 'new': ft_new_acc,
            'per': {k: round(v, 4) for k, v in ft_per.items()},
            'base_ref': base_acc, 'params': n_params, 'forget': round((base_acc - ft_base_acc) * 100, 2)
        }
        print(f'  [OK] Overall: {ft_acc*100:.1f}% | Base: {ft_base_acc*100:.1f}% | New: {ft_new_acc*100:.1f}% | Forget: {(base_acc-ft_base_acc)*100:+.1f}% [{t_ft:.0f}s]')

    # Phase 2b: NC-OPAL v2
    print(f'\n[Phase 2b] NC-OPAL v2 ({bb_name})...')
    t_start = time.time()
    opal_model, opal_intents = ncopal_slu(base_model, BASE_INTENTS, NEW_INTENTS, new_train, base_train_sub)
    opal_acc, opal_per = evaluate_slu(opal_model, opal_intents, all_test_data)
    opal_base_acc = np.mean([opal_per[i] for i in BASE_INTENTS])
    opal_new_acc = np.mean([opal_per[i] for i in NEW_INTENTS])
    t_opal = time.time() - t_start

    n_params = sum(p.numel() for p in bb_fn(n_classes=31).parameters())
    results[f'NC-OPAL ({bb_name})'] = {
        'overall': opal_acc, 'base': opal_base_acc, 'new': opal_new_acc,
        'per': {k: round(v, 4) for k, v in opal_per.items()},
        'base_ref': base_acc, 'params': n_params, 'forget': round((base_acc - opal_base_acc) * 100, 2)
    }
    print(f'  [OK] Overall: {opal_acc*100:.1f}% | Base: {opal_base_acc*100:.1f}% | New: {opal_new_acc*100:.1f}% | Forget: {(base_acc-opal_base_acc)*100:+.1f}% [{t_opal:.0f}s]')

    # Save intermediate results
    elapsed = time.time() - exp_start
    print(f'\n  [Elapsed: {elapsed/60:.1f} min]')

# ==============================
# FINAL SUMMARY
# ==============================
total_time = time.time() - exp_start
print('\n' + '='*74)
print('  NC-SLU RESULTS: 4-WAY BACKBONE COMPARISON (NC-OPAL v2)')
print('='*74)
print(f'{"Method":<28} {"Params":>8} {"Overall":>9} {"Base(25)":>9} {"New(6)":>9} {"Forget":>9}')
print('-'*74)

for name, r in sorted(results.items()):
    print(f'{name:<28} {r["params"]:>7,} {r["overall"]*100:>8.1f}% {r["base"]*100:>8.1f}% {r["new"]*100:>8.1f}% {r["forget"]:>+8.1f}%')

# Best backbone
opal_keys = [k for k in results if k.startswith('NC-OPAL')]
if opal_keys:
    accs = {k: results[k]['overall'] for k in opal_keys}
    best = max(accs, key=accs.get)
    print(f'\n* Best backbone: {best} ({accs[best]*100:.1f}%)')

    print(f'\n--- Per-intent accuracy ({best}) ---')
    best_per = results[best]['per']
    print('  NEW intents:')
    for intent in NEW_INTENTS:
        acc = best_per.get(intent, 0)
        print(f'    {intent}: {acc*100:.1f}%')

print(f'\nTotal time: {total_time/60:.1f} minutes')

# ---- Save results to JSON ----
save_data = {
    'experiment': 'NC-SLU 4-Way Backbone Comparison',
    'dataset': 'Fluent Speech Commands (31 intents)',
    'protocol': '25 base + 6 new (20-shot)',
    'ncopal_version': 'v2',
    'hyperparameters': {
        'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
        'lambda_kd': LAMBDA_KD, 'kd_temp': KD_TEMP,
        'rehearsal_per_class': REHEARSAL_PER_CLASS,
        'old_aug_factor': OLD_AUG_FACTOR, 'new_aug_factor': NEW_AUG_FACTOR,
        'stage1_epochs': STAGE1_EPOCHS, 'stage2_epochs': STAGE2_EPOCHS,
        'batch_size': BATCH_SIZE
    },
    'base_accuracies': {k: round(v, 4) for k, v in base_accs.items()},
    'results': {}
}
for name, r in results.items():
    save_data['results'][name] = {k: v for k, v in r.items() if k != 'per'}
    save_data['results'][name]['per_intent'] = r['per']

save_path = os.path.join(REPO, 'results', 'slu_fsc_v2.json')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f'\n[OK] Results saved to {save_path}')
print('Done!')
