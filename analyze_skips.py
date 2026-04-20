"""
analyze_skips.py
================
Interpretability analysis: are skip connections doing all the work?

Three complementary analyses on a trained AutoEncoder1D (use_skips=True):

  1. Skip Ablation
     Run inference with skip connections zeroed out at each decoder stage
     (independently and all-at-once).  Performance drop = how much each
     skip contributes.  If val_corr barely changes → bottleneck is sufficient.

  2. Activation Norm Ratio
     At each skip injection point, measure
         ratio = ‖skip‖ / (‖skip‖ + ‖bottleneck_upsampled‖)
     ratio ≈ 1 → skips dominate;  ratio ≈ 0 → bottleneck dominates.

  3. Gradient Magnitude
     Backprop a loss signal through the model.  At each skip injection point,
     compare ‖∂L/∂skip‖ vs ‖∂L/∂bottleneck_upsampled‖.
     High gradient through skips → model relies on skip features.

Usage:
  python analyze_skips.py              # analyse latest checkpoint, all subjects
  python analyze_skips.py --subj 1     # subject 1 only
  python analyze_skips.py --version 3  # specific version
"""

import argparse, pathlib, re
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage

from models      import AutoEncoder1D
from train_utils import predict_full, compute_val_corr, SMOOTH_SIGMA

PROJECT_ROOT   = pathlib.Path(__file__).parent
CLEAN_DATA_DIR = PROJECT_ROOT / 'cleaned_data'
CKPT_DIR       = PROJECT_ROOT / 'checkpoints'

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')

WINDOW  = 256
N_FREQS = 40


# ─────────────────────────────────────────────────────────────────────────────
# Model with hooks for ablation / norm tracking
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedAutoEncoder(AutoEncoder1D):
    """
    AutoEncoder1D with per-skip-level controls:
      ablate_levels : set of decoder level indices to zero-out (0 = shallowest)
      record_norms  : if True, record activation norms at each skip injection
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablate_levels = set()
        self.record_norms  = False
        self.norm_log      = []   # list of dicts per forward pass
        self._grad_log     = []   # filled by gradient hooks

    def forward(self, x):
        B, C, n_f, T = x.shape
        x = x.reshape(B, C * n_f, T)
        x = self.spatial_reduce(x)

        skips = []
        for block in self.encoder:
            skips.append(x)
            x = block(x)

        entry = {}
        for i, block in enumerate(self.decoder):
            x_up = block(x)          # upsampled bottleneck path
            skip = skips[-(i + 1)]   # skip from encoder

            t = min(x_up.shape[-1], skip.shape[-1])
            x_up_t  = x_up[..., :t]
            skip_t  = skip[..., :t]

            if self.record_norms:
                up_norm   = x_up_t.detach().norm(dim=(1, 2)).mean().item()
                skip_norm = skip_t.detach().norm(dim=(1, 2)).mean().item()
                entry[f'level_{i}_up_norm']   = up_norm
                entry[f'level_{i}_skip_norm'] = skip_norm
                entry[f'level_{i}_ratio']     = (skip_norm /
                                                  (skip_norm + up_norm + 1e-8))

            # Ablation: replace skip with zeros
            if i in self.ablate_levels:
                skip_t = torch.zeros_like(skip_t)

            x = torch.cat([x_up_t, skip_t], dim=1)

        if self.record_norms:
            self.norm_log.append(entry)

        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_ckpt(stem: str, version: int = None) -> pathlib.Path:
    if version is not None:
        p = CKPT_DIR / f'{stem}_v{version}.pt'
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    matches = list(CKPT_DIR.glob(f'{stem}_v*.pt'))
    if not matches:
        raise FileNotFoundError(f'No checkpoint matching {stem}_v*.pt')
    return max(matches, key=lambda p: int(re.search(r'_v(\d+)\.pt$', p.name).group(1)))


def load_instrumented(n_electrodes: int, subj: int, version: int = None):
    ckpt  = _latest_ckpt(f'subj{subj}_cnn_best', version)
    model = InstrumentedAutoEncoder(n_electrodes=n_electrodes, n_freqs=N_FREQS, n_out=5)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f'  Loaded {ckpt.name}')
    return model


def _val_corr(model, specs_val, y_val):
    corr, per_finger = compute_val_corr(
        model, specs_val, y_val, DEVICE, WINDOW,
        stride_multiple=model.stride_multiple, flat=False,
    )
    return corr, per_finger


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 – Skip Ablation
# ─────────────────────────────────────────────────────────────────────────────

def ablation_analysis(model, specs_val, y_val):
    depth = model.depth   # number of skip levels

    print('\n  [Ablation]')
    base_corr, _ = _val_corr(model, specs_val, y_val)
    print(f'    Baseline (all skips active):  val_corr = {base_corr:.4f}')

    # Zero all skips at once
    model.ablate_levels = set(range(depth))
    no_skip_corr, _ = _val_corr(model, specs_val, y_val)
    print(f'    No skips at all:              val_corr = {no_skip_corr:.4f}'
          f'  (Δ = {no_skip_corr - base_corr:+.4f})')
    model.ablate_levels = set()

    # Zero one skip level at a time
    print(f'    Per-level ablation (0 = shallowest encoder output):')
    for lvl in range(depth):
        model.ablate_levels = {lvl}
        c, _ = _val_corr(model, specs_val, y_val)
        print(f'      Level {lvl}: val_corr = {c:.4f}  (Δ = {c - base_corr:+.4f})')
    model.ablate_levels = set()

    return base_corr, no_skip_corr


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 – Activation Norm Ratio
# ─────────────────────────────────────────────────────────────────────────────

def norm_ratio_analysis(model, specs_val):
    depth = model.depth
    model.record_norms = True
    model.norm_log     = []

    # Run a few windows
    T = specs_val.shape[2]
    n_windows = min(50, T // WINDOW)
    with torch.no_grad():
        for i in range(n_windows):
            x = torch.from_numpy(
                specs_val[..., i * WINDOW:(i + 1) * WINDOW].astype('float32')
            ).unsqueeze(0).to(DEVICE)
            model(x)

    model.record_norms = False

    print('\n  [Activation Norm Ratios]  (ratio = skip / (skip + bottleneck))')
    print(f'    {"Level":<8} {"Skip norm":>10} {"Up norm":>10} {"Ratio":>8}  Interpretation')
    for lvl in range(depth):
        ratios   = [e[f'level_{lvl}_ratio']     for e in model.norm_log]
        skips    = [e[f'level_{lvl}_skip_norm'] for e in model.norm_log]
        ups      = [e[f'level_{lvl}_up_norm']   for e in model.norm_log]
        r   = np.mean(ratios)
        sn  = np.mean(skips)
        un  = np.mean(ups)
        if r > 0.7:
            note = '← skips dominate'
        elif r < 0.3:
            note = '← bottleneck dominates'
        else:
            note = '← balanced'
        print(f'    {lvl:<8} {sn:>10.3f} {un:>10.3f} {r:>8.3f}  {note}')

    return model.norm_log


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 – Gradient Magnitude
# ─────────────────────────────────────────────────────────────────────────────

def gradient_analysis(model, specs_val, y_val):
    """
    For a batch of windows, record the gradient magnitude flowing through
    the skip path vs. the bottleneck-upsampled path at each decoder level.
    Uses register_forward_hook + register_full_backward_hook.
    """
    depth = model.depth

    # Storage: keyed by level
    skip_acts  = {}   # level → tensor saved in forward
    up_acts    = {}   # level → tensor saved in forward
    skip_grads = {i: [] for i in range(depth)}
    up_grads   = {i: [] for i in range(depth)}

    # Monkey-patch forward to attach hooks per call
    original_forward = InstrumentedAutoEncoder.forward

    def instrumented_forward(self, x):
        B, C, n_f, T = x.shape
        x = x.reshape(B, C * n_f, T)
        x = self.spatial_reduce(x)

        skips = []
        for block in self.encoder:
            skips.append(x)
            x = block(x)

        for i, block in enumerate(self.decoder):
            x_up  = block(x)
            skip  = skips[-(i + 1)]
            t     = min(x_up.shape[-1], skip.shape[-1])
            x_up_t  = x_up[..., :t].requires_grad_(True); x_up_t.retain_grad()
            skip_t  = skip[..., :t].requires_grad_(True); skip_t.retain_grad()

            skip_acts[i] = skip_t
            up_acts[i]   = x_up_t

            x = torch.cat([x_up_t, skip_t], dim=1)

        return self.head(x)

    InstrumentedAutoEncoder.forward = instrumented_forward

    T = specs_val.shape[2]
    n_windows = min(20, T // WINDOW)
    C = specs_val.shape[0]

    model.train()   # need grads
    for i in range(n_windows):
        x = torch.from_numpy(
            specs_val[..., i * WINDOW:(i + 1) * WINDOW].astype('float32')
        ).unsqueeze(0).to(DEVICE)
        y = torch.from_numpy(
            y_val[i * WINDOW:(i + 1) * WINDOW].astype('float32')
        ).unsqueeze(0).to(DEVICE)   # (1, W, 5) → need (1, 5, W)
        y = y.permute(0, 2, 1)

        out  = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()

        for lvl in range(depth):
            if skip_acts.get(lvl) is not None and skip_acts[lvl].grad is not None:
                skip_grads[lvl].append(skip_acts[lvl].grad.norm().item())
            if up_acts.get(lvl) is not None and up_acts[lvl].grad is not None:
                up_grads[lvl].append(up_acts[lvl].grad.norm().item())

        model.zero_grad()

    model.eval()
    InstrumentedAutoEncoder.forward = original_forward   # restore

    print('\n  [Gradient Magnitudes]  (‖∂L/∂activation‖)')
    print(f'    {"Level":<8} {"Skip grad":>10} {"Up grad":>10} {"Ratio":>8}  Interpretation')
    for lvl in range(depth):
        sg = np.mean(skip_grads[lvl]) if skip_grads[lvl] else float('nan')
        ug = np.mean(up_grads[lvl])   if up_grads[lvl]   else float('nan')
        if sg + ug > 0:
            r = sg / (sg + ug)
        else:
            r = float('nan')
        if r > 0.7:
            note = '← gradients flow mainly through skips'
        elif r < 0.3:
            note = '← gradients flow mainly through bottleneck'
        else:
            note = '← balanced gradient flow'
        print(f'    {lvl:<8} {sg:>10.4f} {ug:>10.4f} {r:>8.3f}  {note}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subj',    type=int, default=None,
                   help='Subject to analyse (1-indexed); default = all')
    p.add_argument('--version', type=int, default=None,
                   help='Checkpoint version; default = latest')
    args = p.parse_args()

    subjects = [args.subj] if args.subj else [1, 2, 3]

    for subj in subjects:
        d         = CLEAN_DATA_DIR / f'subj{subj}'
        specs_all = np.load(d / 'specs_train.npy')
        y_all     = np.load(d / 'y_train.npy')
        C         = specs_all.shape[0]

        T_cut     = int(specs_all.shape[2] * 0.8)
        specs_val = specs_all[..., T_cut:]
        y_val     = y_all[T_cut:]

        print(f'\n{"="*60}')
        print(f' Subject {subj}')
        print(f'{"="*60}')

        model = load_instrumented(C, subj, args.version)

        ablation_analysis(model,  specs_val, y_val)
        norm_ratio_analysis(model, specs_val)
        gradient_analysis(model,  specs_val, y_val)

    print('\nDone.')


if __name__ == '__main__':
    main()
