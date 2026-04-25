"""
predict_utils.py
================
End-to-end inference pipeline for the CNN v4 finger-flex model.
Imported by predict_test.ipynb; not intended to be run directly.
"""

import pickle
import numpy as np
import scipy.io
import scipy.ndimage
import scipy.interpolate
import torch

from models import AutoEncoder1D
from data_processing import (
    reshape_ecog, normalize_ecog, filter_ecog,
    compute_spectrograms, downsample_spectrograms,
)
from train_utils import predict_full, SMOOTH_SIGMA

N_SUBJECTS         = 3
N_FREQS            = 40
FS                 = 1000
MODEL_HZ           = 100
TARGET_HZ          = 1000
UPSAMPLE_FACTOR    = TARGET_HZ // MODEL_HZ
WINDOW             = 256
TIME_DELAY_SAMPLES = 20


def preprocess_ecog(ecog_raw, ecog_scaler, n_channels, n_freqs=N_FREQS, fs=FS):
    """Raw ECoG (N, C) → scaled spectrogram (C, F, T) at 100 Hz."""
    N          = ecog_raw.shape[0]
    T_expected = N // 10
    ecog  = filter_ecog(normalize_ecog(reshape_ecog(ecog_raw)), fs=fs)
    specs = downsample_spectrograms(compute_spectrograms(ecog, fs=fs))[..., :T_expected]
    n_feats = n_channels * n_freqs
    return (ecog_scaler
            .transform(specs.T.reshape(-1, n_feats))
            .reshape(T_expected, n_freqs, n_channels)
            .T.astype('float32'))


def correct_time_delay(pred_100hz, delay=TIME_DELAY_SAMPLES):
    """Shift predictions back by `delay` samples to undo the training-time offset."""
    T           = pred_100hz.shape[0]
    out         = np.empty_like(pred_100hz)
    out[delay:] = pred_100hz[:T - delay]
    out[:delay] = pred_100hz[0]
    return out


def upsample_to_1000hz(pred_100hz):
    """Cubic interpolation (T, 5) @ 100 Hz → (T*10, 5) @ 1000 Hz."""
    T     = pred_100hz.shape[0]
    t_in  = np.arange(T)
    t_out = np.linspace(0, T - 1, T * UPSAMPLE_FACTOR)
    return np.column_stack([
        scipy.interpolate.interp1d(t_in, pred_100hz[:, f], kind='cubic')(t_out)
        for f in range(5)
    ]).astype(np.float32)


def run_pipeline(input_mat, output_mat, data_key='truetest_data',
                 ckpt_pattern='checkpoints/subj{}_cnn_best_v4.pt',
                 scaler_pattern='scalers/subj{}_ecog_scaler.pkl'):
    device = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    ecog_all     = scipy.io.loadmat(input_mat)[data_key]
    predicted_dg = np.empty((N_SUBJECTS, 1), dtype=object)

    for subj in range(N_SUBJECTS):
        print(f'\nSubject {subj + 1}')
        ecog_raw = ecog_all[subj, 0]
        C        = ecog_raw.shape[1]

        with open(scaler_pattern.format(subj + 1), 'rb') as f:
            ecog_scaler = pickle.load(f)

        print(f'  Preprocessing {ecog_raw.shape} ...', flush=True)
        specs = preprocess_ecog(ecog_raw, ecog_scaler, C)
        print(f'  Spectrogram: {specs.shape}')

        model = AutoEncoder1D(n_electrodes=C, n_freqs=N_FREQS, n_out=5)
        model.load_state_dict(torch.load(ckpt_pattern.format(subj + 1), map_location=device))
        model.to(device).eval()
        print(f'  Loaded {ckpt_pattern.format(subj + 1)}')

        print('  Running inference ...', flush=True)
        raw_pred = predict_full(model, specs, device, WINDOW)
        smooth   = scipy.ndimage.gaussian_filter1d(raw_pred.T, sigma=SMOOTH_SIGMA).T
        pred_1k  = upsample_to_1000hz(correct_time_delay(smooth))
        print(f'  Output shape: {pred_1k.shape}')

        predicted_dg[subj, 0] = pred_1k

    scipy.io.savemat(output_mat, {'predicted_dg': predicted_dg})
    print(f'\nSaved → {output_mat}')
