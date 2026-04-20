"""
train_utils.py
==============
Shared Dataset, DataLoader factory, loss, and evaluation utilities
used by all training scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import scipy.ndimage

EVAL_FINGERS = [0, 1, 2, 4]          # thumb, index, middle, pinky (skip ring)
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
TRAIN_FRAC   = 0.8
SMOOTH_SIGMA = 6                      # Gaussian post-processing σ (samples @ 100 Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(nn.CosineSimilarity(dim=-1)(y_hat, y))


def combined_loss(y_hat: torch.Tensor, y: torch.Tensor,
                  finger_mask: torch.Tensor = None) -> torch.Tensor:
    """0.5*MSE + 0.5*(1 − cosine_sim), optionally restricted to finger_mask."""
    if finger_mask is not None:
        y_hat = y_hat[:, finger_mask, :]
        y     = y[:,     finger_mask, :]
    mse  = F.mse_loss(y_hat, y)
    corr = cosine_sim(y_hat, y)
    return 0.5 * mse + 0.5 * (1.0 - corr)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Sliding-window dataset over (C, F, T) spectrograms.

    Returns:
      x : (C, F, window)   if flat=False  [for CNN]
          (window, C*F)    if flat=True   [for LSTM / Transformer]
      y : (5, window)
    """
    def __init__(self, specs: np.ndarray, y: np.ndarray,
                 window: int, stride: int = 1, flat: bool = False):
        T = specs.shape[2]
        assert y.shape[0] == T, f'specs T={T} but y T={y.shape[0]}'
        self.specs  = torch.from_numpy(specs.astype('float32'))
        self.y      = torch.from_numpy(y.T.astype('float32'))  # (5, T)
        self.window = window
        self.stride = stride
        self.flat   = flat
        self.starts = list(range(0, T - window, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.window
        x = self.specs[..., s:e]                    # (C, F, window)
        if self.flat:
            C, n_f, W = x.shape
            x = x.reshape(C * n_f, W).T             # (window, C*F)
        return x, self.y[:, s:e]                    # y: (5, window)


def make_loaders(specs: np.ndarray, y: np.ndarray,
                 window: int, batch_size: int,
                 train_frac: float = TRAIN_FRAC,
                 train_stride: int = 1,
                 flat: bool = False,
                 num_workers: int = 0):
    """Chronological split → (train_loader, val_loader)."""
    T   = specs.shape[2]
    cut = int(T * train_frac)

    train_ds = WindowDataset(specs[..., :cut], y[:cut],  window, stride=train_stride, flat=flat)
    val_ds   = WindowDataset(specs[..., cut:], y[cut:],  window, stride=window,       flat=flat)

    print(f'  Train windows: {len(train_ds):>6}  Val windows: {len(val_ds):>6}  '
          f'(T={T}, split at t={cut})')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Inference on full sequence
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_full(model: nn.Module, specs: np.ndarray,
                 device: str, window: int,
                 stride_multiple: int = 1,
                 flat: bool = False) -> np.ndarray:
    """
    Run model over a full spectrogram in non-overlapping windows.

    specs : (C, F, T)
    Returns (T, 5) predictions (unsmoothed).
    """
    model.eval()
    C, n_f, T = specs.shape
    T_trim = (T // max(stride_multiple, 1)) * max(stride_multiple, 1)

    preds = []
    step  = window
    for s in range(0, T_trim, step):
        e    = s + step
        chunk = specs[..., s:min(e, T_trim)]

        # Pad last chunk if shorter than window
        if chunk.shape[-1] < window:
            pad   = window - chunk.shape[-1]
            keep  = chunk.shape[-1]
            chunk = np.concatenate([chunk, np.repeat(chunk[..., -1:], pad, axis=-1)], axis=-1)
        else:
            keep = step

        if flat:
            x = torch.from_numpy(chunk.reshape(C * n_f, window).T.astype('float32')).unsqueeze(0).to(device)
        else:
            x = torch.from_numpy(chunk.astype('float32')).unsqueeze(0).to(device)

        out = model(x)[0].cpu().numpy()    # (5, window)
        preds.append(out[:, :keep])

    pred = np.concatenate(preds, axis=1).T   # (T_trim, 5)

    # Repeat last value to fill trimmed tail
    if T_trim < T:
        tail = np.repeat(pred[-1:], T - T_trim, axis=0)
        pred = np.concatenate([pred, tail], axis=0)

    return pred.astype('float32')   # (T, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Validation correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_val_corr(model: nn.Module, specs_val: np.ndarray, y_val: np.ndarray,
                     device: str, window: int,
                     stride_multiple: int = 1,
                     flat: bool = False) -> tuple:
    """
    Predict full validation segment, Gaussian-smooth, compute Pearson r.

    Returns (mean_corr_over_eval_fingers, list_of_5_corrs).
    """
    preds  = predict_full(model, specs_val, device, window, stride_multiple, flat)
    smooth = scipy.ndimage.gaussian_filter1d(preds.T, sigma=SMOOTH_SIGMA).T   # (T, 5)
    T      = min(smooth.shape[0], y_val.shape[0])
    corrs  = [pearsonr(y_val[:T, f], smooth[:T, f])[0] for f in range(5)]
    mean_r = float(np.mean([corrs[f] for f in EVAL_FINGERS]))
    return mean_r, corrs
