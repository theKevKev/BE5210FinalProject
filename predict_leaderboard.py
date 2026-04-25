"""
predict_leaderboard.py
======================
Load trained CNN models and generate leaderboard .mat submissions.

Usage:
  python predict_leaderboard.py             # latest checkpoint
  python predict_leaderboard.py --version 4 # specific version
"""

import argparse, json, pathlib, re
import numpy as np
import scipy.io
import scipy.ndimage
import scipy.interpolate
import torch

from models      import AutoEncoder1D
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
N_FREQS            = 40
TIME_DELAY_SAMPLES = 20     # 200 ms × 100 Hz
N_SUBJECTS         = 3
UPSAMPLE_FACTOR    = 10     # 100 Hz → 1000 Hz


def _latest_ckpt(stem: str, version: int = None) -> pathlib.Path:
    if version is not None:
        p = CKPT_DIR / f'{stem}_v{version}.pt'
        if not p.exists():
            raise FileNotFoundError(f'No checkpoint at {p}')
        return p
    matches = list(CKPT_DIR.glob(f'{stem}_v*.pt'))
    if not matches:
        raise FileNotFoundError(f'No checkpoint matching {stem}_v*.pt')
    return max(matches, key=lambda p: int(re.search(r'_v(\d+)\.pt$', p.name).group(1)))


def _ckpt_version(ckpt: pathlib.Path) -> int:
    return int(re.search(r'_v(\d+)\.pt$', ckpt.name).group(1))


def _skip_levels_for_version(version: int):
    meta_path = META_DIR / f'cnn_v{version}.json'
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    levels = meta.get('architecture', {}).get('skip_levels')
    return set(levels) if levels is not None else None


def correct_time_delay(pred_100hz: np.ndarray,
                       delay: int = TIME_DELAY_SAMPLES) -> np.ndarray:
    T = pred_100hz.shape[0]
    out          = np.empty_like(pred_100hz)
    out[delay:]  = pred_100hz[:T - delay]
    out[:delay]  = pred_100hz[0]
    return out


def upsample_to_1000hz(pred_100hz: np.ndarray) -> np.ndarray:
    T     = pred_100hz.shape[0]
    t_in  = np.arange(T)
    t_out = np.linspace(0, T - 1, T * UPSAMPLE_FACTOR)
    return np.column_stack([
        scipy.interpolate.interp1d(t_in, pred_100hz[:, f], kind='cubic')(t_out)
        for f in range(5)
    ]).astype(np.float32)


def generate_submission(version: int = None):
    probe_ckpt  = _latest_ckpt('subj1_cnn_best', version)
    run_version = version if version is not None else _ckpt_version(probe_ckpt)
    skip_levels = _skip_levels_for_version(run_version)

    print(f'\nGenerating CNN v{run_version} predictions')
    predicted_dg = np.empty((N_SUBJECTS, 1), dtype=object)

    for subj in range(N_SUBJECTS):
        d          = CLEAN_DATA_DIR / f'subj{subj + 1}'
        specs_lead = np.load(d / 'specs_lead.npy')
        C          = specs_lead.shape[0]

        ckpt  = _latest_ckpt(f'subj{subj + 1}_cnn_best', run_version)
        model = AutoEncoder1D(n_electrodes=C, n_freqs=N_FREQS, n_out=5,
                              skip_levels=skip_levels)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()
        print(f'  Subject {subj + 1}  specs_lead={specs_lead.shape}  loaded {ckpt.name}')

        raw_pred    = predict_full(model, specs_lead, DEVICE, WINDOW,
                                   stride_multiple=model.stride_multiple)
        smooth      = scipy.ndimage.gaussian_filter1d(raw_pred.T, sigma=SMOOTH_SIGMA).T
        pred_1000hz = upsample_to_1000hz(correct_time_delay(smooth))
        print(f'    prediction shape: {pred_1000hz.shape}')

        predicted_dg[subj, 0] = pred_1000hz

    out_path = PRED_DIR / f'leaderboard_cnn_v{run_version}.mat'
    scipy.io.savemat(str(out_path), {'predicted_dg': predicted_dg})
    print(f'  Saved → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', type=int, default=None,
                   help='Checkpoint version (default: latest)')
    args = p.parse_args()
    generate_submission(version=args.version)


if __name__ == '__main__':
    main()
