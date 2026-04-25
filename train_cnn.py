"""
train_cnn.py
============
Supervised fine-tuning of AutoEncoder1D (1D U-Net CNN) for finger flex decoding.

Usage:
  python train_cnn.py
"""

import json, pathlib, re
from datetime import datetime
import numpy as np
import torch

from models     import AutoEncoder1D, _DEFAULT_CHANNELS, _DEFAULT_KERNELS, _DEFAULT_STRIDES, _DEFAULT_DILATIONS
from train_utils import (make_loaders, combined_loss, compute_val_corr,
                          EVAL_FINGERS, FINGER_NAMES)

PROJECT_ROOT   = pathlib.Path(__file__).parent
CLEAN_DATA_DIR = PROJECT_ROOT / 'cleaned_data'
CKPT_DIR       = PROJECT_ROOT / 'checkpoints'
META_DIR       = PROJECT_ROOT / 'metadata'
CKPT_DIR.mkdir(exist_ok=True)
META_DIR.mkdir(exist_ok=True)

SEED = 42

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')

WINDOW          = 256
BATCH_SIZE      = 32
FINETUNE_EPOCHS = 30
LR              = 8.42e-5
TRAIN_STRIDE    = 2
LOSS_FINGERS    = [0, 1, 2, 4]   # skip ring finger
N_SUBJECTS      = 3


def _set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _next_version(stem: str) -> int:
    """Return the next unused version number for a checkpoint stem."""
    existing = list(CKPT_DIR.glob(f'{stem}_v*.pt'))
    if not existing:
        return 1
    nums = [int(re.search(r'_v(\d+)\.pt$', p.name).group(1)) for p in existing]
    return max(nums) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_subject(subj: int, specs_train: np.ndarray, y_train: np.ndarray,
                  run_version: int = 1):
    print(f'\n[Train] Subject {subj + 1}')
    C, n_freqs, T = specs_train.shape

    model = AutoEncoder1D(n_electrodes=C, n_freqs=n_freqs, n_out=5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    train_loader, val_loader = make_loaders(
        specs_train, y_train, WINDOW, BATCH_SIZE, train_stride=TRAIN_STRIDE,
    )

    T_cut     = int(T * 0.8)
    specs_val = specs_train[..., T_cut:]
    y_val     = y_train[T_cut:]

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    ckpt  = CKPT_DIR / f'subj{subj + 1}_cnn_best_v{run_version}.pt'

    best_corr = -1.0
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'val_corr':>8}")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = combined_loss(model(x), y, torch.tensor(LOSS_FINGERS))
            loss.backward()
            opt.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                v_loss += combined_loss(model(x), y, torch.tensor(LOSS_FINGERS)).item()
        v_loss /= max(len(val_loader), 1)

        val_corr, per_finger = compute_val_corr(
            model, specs_val, y_val, DEVICE, WINDOW,
            stride_multiple=model.stride_multiple,
        )
        sched.step()

        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), ckpt)

        if epoch % 5 == 0 or epoch == 1:
            finger_str = '  '.join(f'{FINGER_NAMES[f]}={per_finger[f]:.3f}'
                                   for f in EVAL_FINGERS)
            print(f'  {epoch:5d}  {t_loss:10.4f}  {v_loss:8.4f}  {val_corr:8.4f}  [{finger_str}]')

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val_corr = {best_corr:.4f}  →  {ckpt.name}')
    return model, best_corr


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(run_version: int, summary: dict):
    meta = {
        'run_version':      run_version,
        'timestamp':        datetime.now().isoformat(timespec='seconds'),
        'seed':             SEED,
        'device':           DEVICE,
        'hyperparameters': {
            'window':          WINDOW,
            'batch_size':      BATCH_SIZE,
            'finetune_epochs': FINETUNE_EPOCHS,
            'lr':              LR,
            'train_stride':    TRAIN_STRIDE,
            'loss_fingers':    LOSS_FINGERS,
        },
        'architecture': {
            'channels':    _DEFAULT_CHANNELS,
            'kernels':     _DEFAULT_KERNELS,
            'strides':     _DEFAULT_STRIDES,
            'dilations':   _DEFAULT_DILATIONS,
            'skip_levels': list(range(len(_DEFAULT_STRIDES))),   # all levels
        },
        'results': {f'subj{s}_val_corr': round(c, 4) for s, c in summary.items()},
        'mean_val_corr': round(float(np.mean(list(summary.values()))), 4),
    }
    out = META_DIR / f'cnn_v{run_version}.json'
    with open(out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  Metadata saved → {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _set_seed(SEED)
    print(f'Device: {DEVICE}  |  Seed: {SEED}\n')

    run_version = _next_version('subj1_cnn_best')
    print(f'Run version: v{run_version}')

    summary = {}
    for subj in range(N_SUBJECTS):
        d     = CLEAN_DATA_DIR / f'subj{subj + 1}'
        specs = np.load(d / 'specs_train.npy')
        y     = np.load(d / 'y_train.npy')

        _, corr = train_subject(subj, specs, y, run_version=run_version)
        summary[subj + 1] = corr

    print('\n=== Summary ===')
    for s, c in summary.items():
        print(f'  Subject {s}: val_corr = {c:.4f}')
    print(f'  Mean: {np.mean(list(summary.values())):.4f}')

    save_metadata(run_version, summary)


if __name__ == '__main__':
    main()
