"""
train_cnn.py
============
Two-phase training for AutoEncoder1D:

  Phase 1 — Self-supervised pre-training (reconstruction).
    Train the U-Net to reconstruct its own input spectrograms.
    Uses ALL available ECoG (training + leaderboard) — no labels needed.
    This lets the encoder learn general neural representations before
    seeing any finger-flex labels.

  Phase 2 — Supervised fine-tuning.
    Swap the reconstruction head for a 5-finger prediction head.
    Copy all encoder/decoder weights from Phase 1; only the final 1×1
    conv is reinitialised.  Fine-tune on labelled training data.

Usage:
  python train_cnn.py                  # pre-train then fine-tune
  python train_cnn.py --no-pretrain    # fine-tune from random init
"""

import argparse, json, pathlib, re
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

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
PRETRAIN_EPOCHS = 20
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
# Phase 1 – Pre-training (reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_subject(subj: int, specs_train: np.ndarray, specs_lead: np.ndarray):
    print(f'\n[Pretrain] Subject {subj + 1}')
    C, n_freqs, T = specs_train.shape
    n_out   = C * n_freqs                  # reconstruct flattened input

    # Combine train + leaderboard — no labels needed
    specs_all = np.concatenate([specs_train, specs_lead], axis=2)
    T_all = specs_all.shape[2]
    T_cut = int(T_all * 0.9)

    model = AutoEncoder1D(n_electrodes=C, n_freqs=n_freqs, n_out=n_out, use_skips=False).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

    n_train_windows = 300

    best_r2 = -float('inf')
    v    = _next_version(f'subj{subj + 1}_pretrained')
    ckpt = CKPT_DIR / f'subj{subj + 1}_pretrained_v{v}.pt'

    print(f"  {'Epoch':>5}  {'train_mse':>9}  {'val_mse':>7}  {'val_r2':>6}")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        starts  = np.random.randint(0, T_cut - WINDOW, size=n_train_windows)
        t_loss  = 0.0
        for s in starts:
            x = (torch.from_numpy(specs_all[..., s:s + WINDOW].astype('float32'))
                 .unsqueeze(0).to(DEVICE))
            target = x.reshape(x.shape[0], C * n_freqs, WINDOW)
            opt.zero_grad()
            out  = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        t_loss /= n_train_windows

        model.eval()
        val_starts = np.arange(T_cut, T_all - WINDOW, WINDOW)
        v_loss = 0.0
        ss_res = 0.0
        ss_tot = 0.0
        with torch.no_grad():
            for s in val_starts:
                x = (torch.from_numpy(specs_all[..., s:s + WINDOW].astype('float32'))
                     .unsqueeze(0).to(DEVICE))
                target = x.reshape(x.shape[0], C * n_freqs, WINDOW)
                pred   = model(x)
                v_loss += F.mse_loss(pred, target).item()
                ss_res += ((target - pred) ** 2).sum().item()
                ss_tot += ((target - target.mean()) ** 2).sum().item()
        v_loss /= max(len(val_starts), 1)
        v_r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        sched.step()

        if v_r2 > best_r2:
            best_r2 = v_r2
            torch.save(model.state_dict(), ckpt)

        if epoch % 5 == 0 or epoch == 1:
            print(f'  {epoch:5d}  {t_loss:9.4f}  {v_loss:7.4f}  {v_r2:6.3f}')

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val_r2 = {best_r2:.4f}  →  {ckpt.name}')
    return model, best_r2


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Fine-tuning (finger flex prediction)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_subject(subj: int, specs_train: np.ndarray, y_train: np.ndarray,
                     pretrained_state: dict = None, run_version: int = 1,
                     skip_levels=None):
    print(f'\n[Fine-tune] Subject {subj + 1}')
    C, n_freqs, T = specs_train.shape

    model = AutoEncoder1D(n_electrodes=C, n_freqs=n_freqs, n_out=5,
                          skip_levels=skip_levels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    if pretrained_state is not None:
        own = model.state_dict()
        transfer = {k: v for k, v in pretrained_state.items()
                    if k in own and own[k].shape == v.shape and 'head' not in k}
        own.update(transfer)
        model.load_state_dict(own)
        print(f'  Loaded {len(transfer)}/{len(own)} weight tensors from pretrained checkpoint')

    train_loader, val_loader = make_loaders(
        specs_train, y_train, WINDOW, BATCH_SIZE,
        train_stride=TRAIN_STRIDE, flat=False,
    )

    T_cut     = int(T * 0.8)
    specs_val = specs_train[..., T_cut:]
    y_val     = y_train[T_cut:]

    best_corr = -1.0
    ckpt = CKPT_DIR / f'subj{subj + 1}_cnn_best_v{run_version}.pt'

    # When using pretrained weights: freeze encoder for first half of fine-tuning
    # so the decoder can learn to use the pretrained encoder features before
    # everything starts shifting together.
    FREEZE_EPOCHS = FINETUNE_EPOCHS // 2 if pretrained_state is not None else 0
    encoder_params = (list(model.spatial_reduce.parameters()) +
                      list(model.encoder.parameters()))

    def _set_encoder_grad(requires_grad: bool):
        for p in encoder_params:
            p.requires_grad = requires_grad

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)

    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'val_corr':>8}")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        # Freeze encoder for first FREEZE_EPOCHS, then unfreeze
        if epoch == 1 and FREEZE_EPOCHS > 0:
            _set_encoder_grad(False)
            print(f'  Encoder frozen for epochs 1–{FREEZE_EPOCHS}')
        elif epoch == FREEZE_EPOCHS + 1:
            _set_encoder_grad(True)
            print(f'  Encoder unfrozen at epoch {epoch}')

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
            stride_multiple=model.stride_multiple, flat=False,
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

def save_metadata(run_version: int, pretrain: bool, summary: dict,
                  pretrain_r2s: dict, skip_levels=None):
    meta = {
        'run_version':      run_version,
        'timestamp':        datetime.now().isoformat(timespec='seconds'),
        'seed':             SEED,
        'device':           DEVICE,
        'pretrain':         pretrain,
        'hyperparameters': {
            'window':          WINDOW,
            'batch_size':      BATCH_SIZE,
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'finetune_epochs': FINETUNE_EPOCHS,
            'lr':              LR,
            'train_stride':    TRAIN_STRIDE,
            'loss_fingers':    LOSS_FINGERS,
            'freeze_epochs':   FINETUNE_EPOCHS // 2 if pretrain else 0,
        },
        'architecture': {
            'channels':    _DEFAULT_CHANNELS,
            'kernels':     _DEFAULT_KERNELS,
            'strides':     _DEFAULT_STRIDES,
            'dilations':   _DEFAULT_DILATIONS,
            'skip_levels': sorted(skip_levels) if skip_levels is not None else list(range(len(_DEFAULT_STRIDES))),
        },
        'results': {
            f'subj{s}_val_corr':   round(c, 4) for s, c in summary.items()
        },
        'mean_val_corr': round(float(np.mean(list(summary.values()))), 4),
    }
    if pretrain_r2s:
        meta['pretrain_r2s'] = {f'subj{s}': round(r, 4) for s, r in pretrain_r2s.items()}

    out = META_DIR / f'cnn_v{run_version}.json'
    with open(out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  Metadata saved → {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(pretrain: bool = True, skip_levels=None):
    _set_seed(SEED)
    sl_str = 'all' if skip_levels is None else str(sorted(skip_levels))
    print(f'Device: {DEVICE}  |  Seed: {SEED}  |  skip_levels={sl_str}\n')

    # All subjects share the same run version number
    run_version = _next_version('subj1_cnn_best')
    print(f'Run version: v{run_version}')

    summary      = {}
    pretrain_r2s = {}

    for subj in range(N_SUBJECTS):
        d         = CLEAN_DATA_DIR / f'subj{subj + 1}'
        specs     = np.load(d / 'specs_train.npy')
        y         = np.load(d / 'y_train.npy')
        specs_ld  = np.load(d / 'specs_lead.npy')

        pretrained_state = None
        if pretrain:
            pm, best_r2 = pretrain_subject(subj, specs, specs_ld)
            pretrained_state = pm.state_dict()
            pretrain_r2s[subj + 1] = best_r2

        model, corr = finetune_subject(subj, specs, y, pretrained_state,
                                       run_version=run_version,
                                       skip_levels=skip_levels)
        summary[subj + 1] = corr

    print('\n=== CNN Summary ===')
    for s, c in summary.items():
        print(f'  Subject {s}: val_corr = {c:.4f}')
    print(f'  Mean: {np.mean(list(summary.values())):.4f}')

    save_metadata(run_version, pretrain, summary, pretrain_r2s, skip_levels)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--no-pretrain', action='store_true',
                   help='Skip self-supervised pre-training')
    p.add_argument('--skip-levels', type=int, nargs='*', default=None,
                   help='Decoder step indices that receive skip connections '
                        '(default: all). E.g. --skip-levels 0 1')
    args = p.parse_args()
    skip_levels = set(args.skip_levels) if args.skip_levels is not None else None
    main(pretrain=not args.no_pretrain, skip_levels=skip_levels)
