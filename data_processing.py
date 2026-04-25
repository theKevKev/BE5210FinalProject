"""
ECoG data preprocessing pipeline for BCI finger flexion prediction.

Features per channel: 40 Morlet wavelet power bands, log-spaced 40–300 Hz.
"""

import numpy as np
import mne
import scipy.signal
import scipy.interpolate
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
L_FREQ          = 40     # Bandpass lower bound (Hz)
H_FREQ          = 300    # Bandpass upper bound (Hz)
WAVELET_LOW     = 40     # Lowest Morlet wavelet frequency (Hz)
WAVELET_HIGH    = 300    # Highest Morlet wavelet frequency (Hz)
WAVELET_NUM     = 40     # Number of log-spaced Morlet frequencies
DOWNSAMPLE_FS   = 100    # Target sampling rate after downsampling (Hz)
TIME_DELAY_SECS = 0.20   # Neural-to-motor time delay (seconds)
DATAGLOVE_FS    = 25     # True dataglove sampling rate (stored as ZOH at 1000 Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Reshape
# ─────────────────────────────────────────────────────────────────────────────
def reshape_ecog(ecog: np.ndarray) -> np.ndarray:
    return ecog.T   # (samples, channels) → (channels, samples)


def reshape_fingerflex(finger_flex: np.ndarray) -> np.ndarray:
    return finger_flex.T   # (samples, fingers) → (fingers, samples)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Normalize ECoG
# ─────────────────────────────────────────────────────────────────────────────
def normalize_ecog(multichannel_signal: np.ndarray) -> np.ndarray:
    """Z-score each channel then subtract cross-channel median (CAR)."""
    means = np.mean(multichannel_signal, axis=1, keepdims=True)
    stds  = np.std(multichannel_signal,  axis=1, keepdims=True)
    norm  = (multichannel_signal - means) / stds
    return norm - np.median(norm, axis=0, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Filter ECoG
# ─────────────────────────────────────────────────────────────────────────────
def filter_ecog(multichannel_signal: np.ndarray,
                fs: int = 1000,
                powerline_freq: int = 60) -> np.ndarray:
    """Bandpass [L_FREQ, H_FREQ] Hz then notch at powerline harmonics."""
    harmonics = np.array([
        i * powerline_freq
        for i in range(1, (fs // 2) // powerline_freq)
    ])
    filtered = mne.filter.filter_data(
        multichannel_signal, fs, l_freq=L_FREQ, h_freq=H_FREQ
    )
    return mne.filter.notch_filter(filtered, fs, freqs=harmonics)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Morlet wavelet spectrograms  (C, 40, T_new)
# ─────────────────────────────────────────────────────────────────────────────
def compute_spectrograms(multichannel_signal: np.ndarray,
                         fs: int = 1000,
                         freqs: np.ndarray = None) -> np.ndarray:
    """
    Morlet wavelet power, log-spaced from WAVELET_LOW to WAVELET_HIGH Hz.

    Input:  (C, T) at fs Hz
    Output: (C, WAVELET_NUM, T) at fs Hz  — downsample separately
    """
    if freqs is None:
        freqs = np.logspace(np.log10(WAVELET_LOW), np.log10(WAVELET_HIGH), WAVELET_NUM)

    n_channels   = multichannel_signal.shape[0]
    spectrograms = mne.time_frequency.tfr_array_morlet(
        multichannel_signal.reshape(1, n_channels, -1),
        sfreq=fs, freqs=freqs, output='power', verbose=False, n_jobs=-1,
    )[0]
    return spectrograms


def downsample_spectrograms(spectrograms: np.ndarray,
                            cur_fs: int = 1000,
                            new_fs: int = DOWNSAMPLE_FS) -> np.ndarray:
    """Decimate along time axis from cur_fs to new_fs."""
    step = cur_fs // new_fs
    return spectrograms[:, :, ::step]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Downsample finger flex labels
# ─────────────────────────────────────────────────────────────────────────────
def downsample_fingerflex(finger_flex: np.ndarray,
                          cur_fs: int = 1000,
                          new_fs: int = DOWNSAMPLE_FS,
                          dataglove_fs: int = DATAGLOVE_FS) -> np.ndarray:
    """
    Cubic interpolation from true 25 Hz → new_fs Hz.
    Avoids the step-function artefact of simple striding.
    """
    zoh_step = cur_fs // dataglove_fs
    out_step = cur_fs // new_fs
    ff_25hz  = finger_flex[:, ::zoh_step]
    n_fingers, n_25 = ff_25hz.shape

    t_25  = np.arange(n_25) * zoh_step
    n_out = finger_flex.shape[1] // out_step
    t_out = np.arange(n_out) * out_step

    result = np.zeros((n_fingers, n_out))
    for f in range(n_fingers):
        interp_fn = scipy.interpolate.interp1d(
            t_25, ff_25hz[f], kind='cubic', bounds_error=False,
            fill_value=(ff_25hz[f, 0], ff_25hz[f, -1]),
        )
        result[f] = interp_fn(t_out)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Align with time delay
# ─────────────────────────────────────────────────────────────────────────────
def crop_for_time_delay(finger_flex: np.ndarray,
                        spectrograms: np.ndarray,
                        time_delay_sec: float = TIME_DELAY_SECS,
                        fs: int = DOWNSAMPLE_FS):
    """Pair specs[t] with ff[t + delay], dropping the boundary samples."""
    delay = int(time_delay_sec * fs)
    return finger_flex[:, delay:], spectrograms[:, :, :-delay]


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Scale
# ─────────────────────────────────────────────────────────────────────────────
def scale_fingerflex(train_ff: np.ndarray, *other_ffs):
    scaler      = MinMaxScaler()
    scaler.fit(train_ff.T)
    scaled_train  = scaler.transform(train_ff.T).T
    scaled_others = tuple(scaler.transform(ff.T).T for ff in other_ffs)
    return (scaler, scaled_train) + scaled_others


def scale_ecog(train_ecog: np.ndarray, *other_ecogs):
    n_channels, n_freqs, _ = train_ecog.shape
    n_feats = n_channels * n_freqs

    scaler = RobustScaler(unit_variance=True, quantile_range=(0.1, 0.9))
    scaler.fit(train_ecog.T.reshape(-1, n_feats))

    def _transform(arr):
        _, _, T = arr.shape
        return (scaler.transform(arr.T.reshape(-1, n_feats))
                      .reshape(T, n_freqs, n_channels)
                      .T)

    scaled_train  = _transform(train_ecog)
    scaled_others = tuple(_transform(e) for e in other_ecogs)
    return (scaler, scaled_train) + scaled_others
