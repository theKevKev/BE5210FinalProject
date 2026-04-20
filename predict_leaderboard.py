"""
predict_leaderboard.py
======================
Load trained models and generate leaderboard .mat submissions.

Training used a 300 ms time-delay shift (specs[t] → ff[t+300ms]), so the
model output at position t predicts ff[t+300ms].  We correct this by
shifting predictions BACK by TIME_DELAY_SAMPLES so submission[t] aligns
with leaderboard index t.

Usage:
  python predict_leaderboard.py --model cnn
  python predict_leaderboard.py --model bigru
  python predict_leaderboard.py --model all
"""

import argparse, json, pathlib, re
import numpy as np
import scipy.io
import scipy.ndimage
import scipy.interpolate
import torch

from models      import AutoEncoder1D, BiGRUPredictor
from train_utils import predict_full, SMOOTH_SIGMA

PROJECT_ROOT   = pathlib.Path(__file__).parent
CLEAN_DATA_DIR = PROJECT_ROOT / 'cleaned_data'
CKPT_DIR       = PROJECT_ROOT / 'checkpoints'
PRED_DIR       = PROJECT_ROOT / 'predictions'
META_DIR       = PROJECT_ROOT / 'metadata'
PRED_DIR.mkdir(exist_ok=True)

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')

WINDOW             = 256
N_FREQS            = 40     # Morlet wavelet frequencies
TIME_DELAY_SAMPLES = 20     # 200 ms × 100 Hz
N_SUBJECTS         = 3
TARGET_HZ          = 1000
MODEL_HZ           = 100
UPSAMPLE_FACTOR    = TARGET_HZ // MODEL_HZ   # 10


def _latest_ckpt(stem: str, version: int = None) -> pathlib.Path:
    """Return the checkpoint for a given stem, at a specific version or the latest."""
    if version is not None:
        p = CKPT_DIR / f'{stem}_v{version}.pt'
        if not p.exists():
            raise FileNotFoundError(f'No checkpoint found at {p}')
        return p
    matches = list(CKPT_DIR.glob(f'{stem}_v*.pt'))
    if not matches:
        raise FileNotFoundError(f'No checkpoint found matching {stem}_v*.pt')
    return max(matches, key=lambda p: int(re.search(r'_v(\d+)\.pt$', p.name).group(1)))


def _ckpt_version(ckpt: pathlib.Path) -> int:
    return int(re.search(r'_v(\d+)\.pt$', ckpt.name).group(1))


def _skip_levels_for_version(version: int):
    """Read skip_levels from metadata JSON for a given run version."""
    meta_path = META_DIR / f'cnn_v{version}.json'
    if not meta_path.exists():
        return None   # no metadata → assume all skips (v1-era)
    with open(meta_path) as f:
        meta = json.load(f)
    levels = meta.get('architecture', {}).get('skip_levels')
    return set(levels) if levels is not None else None


def load_model(model_type: str, n_electrodes: int, subj: int,
               version: int = None) -> torch.nn.Module:
    n_features = n_electrodes * N_FREQS
    if model_type == 'cnn':
        skip_levels = _skip_levels_for_version(version) if version is not None else None
        model = AutoEncoder1D(n_electrodes=n_electrodes, n_freqs=N_FREQS, n_out=5,
                              skip_levels=skip_levels)
        ckpt  = _latest_ckpt(f'subj{subj + 1}_cnn_best', version)
    elif model_type == 'bigru':
        model = BiGRUPredictor(n_features)
        ckpt  = _latest_ckpt(f'subj{subj + 1}_bigru_best', version)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f'  Loaded {ckpt.name}')
    return model


def correct_time_delay(pred_100hz: np.ndarray,
                       delay: int = TIME_DELAY_SAMPLES) -> np.ndarray:
    """Shift predictions back by delay samples to undo the training-time offset."""
    T = pred_100hz.shape[0]
    out          = np.empty_like(pred_100hz)
    out[delay:]  = pred_100hz[:T - delay]
    out[:delay]  = pred_100hz[0]
    return out


def upsample_to_1000hz(pred_100hz: np.ndarray) -> np.ndarray:
    """Cubic interpolation from 100 Hz to 1000 Hz.  (T, 5) → (T*10, 5)"""
    T     = pred_100hz.shape[0]
    t_in  = np.arange(T)
    t_out = np.linspace(0, T - 1, T * UPSAMPLE_FACTOR)
    return np.column_stack([
        scipy.interpolate.interp1d(t_in, pred_100hz[:, f], kind='cubic')(t_out)
        for f in range(5)
    ]).astype(np.float32)


def generate_submission(model_type: str, version: int = None):
    print(f'\nGenerating predictions: {model_type.upper()}')
    predicted_dg = np.empty((N_SUBJECTS, 1), dtype=object)

    # Determine run version from subject 1 checkpoint
    probe_stem = f'subj1_{model_type}_best'
    run_version = version if version is not None else _ckpt_version(_latest_ckpt(probe_stem))

    for subj in range(N_SUBJECTS):
        d          = CLEAN_DATA_DIR / f'subj{subj + 1}'
        specs_lead = np.load(d / 'specs_lead.npy')   # (C, F, T_lead)
        C          = specs_lead.shape[0]

        print(f'  Subject {subj + 1}  specs_lead={specs_lead.shape}')
        model = load_model(model_type, C, subj, version=run_version)

        flat     = (model_type != 'cnn')
        sm       = getattr(model, 'stride_multiple', 1)
        raw_pred = predict_full(model, specs_lead, DEVICE, WINDOW,
                                stride_multiple=sm, flat=flat)   # (T_lead, 5)

        smooth      = scipy.ndimage.gaussian_filter1d(raw_pred.T, sigma=SMOOTH_SIGMA).T
        aligned     = correct_time_delay(smooth)
        pred_1000hz = upsample_to_1000hz(aligned)
        print(f'    prediction shape: {pred_1000hz.shape}')

        predicted_dg[subj, 0] = pred_1000hz

    out_path = PRED_DIR / f'leaderboard_{model_type}_v{run_version}.mat'
    scipy.io.savemat(str(out_path), {'predicted_dg': predicted_dg})
    print(f'  Saved → {out_path}')
    return str(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cnn', choices=['cnn', 'bigru', 'all'])
    p.add_argument('--version', type=int, default=None,
                   help='Checkpoint version to use (default: latest)')
    args = p.parse_args()

    models = ['cnn', 'bigru'] if args.model == 'all' else [args.model]
    for m in models:
        try:
            generate_submission(m, version=args.version)
        except FileNotFoundError as e:
            print(f'  Skipping {m}: {e}')


if __name__ == '__main__':
    main()
