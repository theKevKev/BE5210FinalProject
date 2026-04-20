"""
prepare_data.py
===============
Rebuilds cleaned_data/ from raw .mat files.

Saves per subject to cleaned_data/subj{N}/:
  specs_train.npy  (C, F, T)      scaled ECoG spectrograms
  y_train.npy      (T, 5)         scaled finger flex labels  [ML-ready]
  X_train.npy      (T, C*F)       flat spectrogram features  [ML-ready]
  specs_lead.npy   (C, F, T_lead) scaled leaderboard spectrograms
  X_lead.npy       (T_lead, C*F)  flat leaderboard features  [ML-ready]
  ecog_scaler.pkl                 fitted RobustScaler  (for test-set use)
  ff_scaler.pkl                   fitted MinMaxScaler  (for test-set use)

NO time-delay shift is applied. The model learns the temporal relationship
directly from the windowed data (the 200 ms neural lead is within every
2.56 s window). This also keeps leaderboard predictions naturally aligned.
"""

import pathlib, pickle, sys
import numpy as np
import scipy.io
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from data_processing import (
    reshape_ecog, reshape_fingerflex, normalize_ecog, filter_ecog,
    compute_spectrograms, downsample_spectrograms, downsample_fingerflex,
    crop_for_time_delay, scale_fingerflex, scale_ecog,
)

PROJECT_ROOT   = pathlib.Path(__file__).parent
RAW_DATA_DIR   = PROJECT_ROOT / 'raw_data'
CLEAN_DATA_DIR = PROJECT_ROOT / 'cleaned_data'


def process_subject(ecog_tr, dg_tr, ecog_lead, subj_idx, fs=1000):
    C_label = subj_idx + 1
    print(f'  Normalising + filtering...', flush=True)
    e = filter_ecog(normalize_ecog(reshape_ecog(ecog_tr)), fs=fs)

    print(f'  Morlet spectrograms (train)...', flush=True)
    specs = downsample_spectrograms(compute_spectrograms(e, fs=fs))

    print(f'  Morlet spectrograms (leaderboard)...', flush=True)
    el         = filter_ecog(normalize_ecog(reshape_ecog(ecog_lead)), fs=fs)
    specs_lead = downsample_spectrograms(compute_spectrograms(el, fs=fs))

    # Finger flex: cubic interpolation from true 25 Hz → 100 Hz
    ff = downsample_fingerflex(reshape_fingerflex(dg_tr))   # (5, T_ff)

    # 200 ms time-delay shift: specs[t] → ff[t+200ms]
    # At prediction time, output is shifted BACK by 200 ms to align with test indices.
    ff, specs = crop_for_time_delay(ff, specs)   # drops first 20 ff samples + last 20 spec samples

    # Scale: fit scalers on training data, apply to both train and lead
    ff_scaler, ff_scaled = scale_fingerflex(ff)
    ecog_scaler, specs_scaled, specs_lead_scaled = scale_ecog(specs, specs_lead)

    C, F, T = specs_scaled.shape
    T_lead  = specs_lead_scaled.shape[2]

    # Flat features: (T, C*F)
    X_train = specs_scaled.reshape(C * F, T).T.astype('float32')
    X_lead  = specs_lead_scaled.reshape(C * F, T_lead).T.astype('float32')

    # Labels: (T, 5)
    y_train = ff_scaled.T.astype('float32')

    return specs_scaled, y_train, X_train, specs_lead_scaled, X_lead, ecog_scaler, ff_scaler


def main():
    print('Loading raw data...')
    mat_train = scipy.io.loadmat(RAW_DATA_DIR / 'raw_training_data.mat')
    mat_lead  = scipy.io.loadmat(RAW_DATA_DIR / 'leaderboard_data.mat')

    ecog_all = mat_train['train_ecog']      # (3, 1) cell
    dg_all   = mat_train['train_dg']        # (3, 1) cell
    lead_all = mat_lead['leaderboard_ecog'] # (3, 1) cell

    for subj in range(3):
        print(f'\n=== Subject {subj + 1} ===')
        specs, y, X, specs_lead, X_lead, ecog_sc, ff_sc = process_subject(
            ecog_all[subj, 0], dg_all[subj, 0], lead_all[subj, 0], subj,
        )

        # Verify labels are smooth
        zero_frac = (np.diff(y[:, 0]) == 0).mean()
        print(f'  y zero-diff fraction: {zero_frac:.3f}  (should be ~0 for smooth labels)')
        print(f'  specs={specs.shape}  y={y.shape}  X={X.shape}')
        print(f'  specs_lead={specs_lead.shape}  X_lead={X_lead.shape}')

        subj_dir = CLEAN_DATA_DIR / f'subj{subj + 1}'
        subj_dir.mkdir(parents=True, exist_ok=True)

        np.save(subj_dir / 'specs_train.npy', specs)
        np.save(subj_dir / 'y_train.npy',     y)
        np.save(subj_dir / 'X_train.npy',     X)
        np.save(subj_dir / 'specs_lead.npy',  specs_lead)
        np.save(subj_dir / 'X_lead.npy',      X_lead)

        with open(subj_dir / 'ecog_scaler.pkl', 'wb') as f:
            pickle.dump(ecog_sc, f)
        with open(subj_dir / 'ff_scaler.pkl', 'wb') as f:
            pickle.dump(ff_sc, f)

        print(f'  Saved to {subj_dir}')

    print('\nAll subjects done.')


if __name__ == '__main__':
    main()
