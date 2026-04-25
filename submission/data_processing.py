"""
ECoG data preprocessing pipeline for BCI finger flexion prediction.

Features per channel (F = 45 total):
  4  band-envelope amplitudes  — delta(1-4), alpha(8-13), beta(13-30), lo-gamma(30-40) Hz
                                  computed via Hilbert envelope on 1000 Hz signal
  40 Morlet wavelet power bands — log-spaced 40-200 Hz
  1  line length               — sliding-window sum of |Δsignal| (100 ms window)
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
LL_WINDOW_MS    = 100    # Line-length sliding window (ms)

# Low-frequency band envelope definitions
BAND_ENVELOPES = [
    (1,  4,  'delta'),
    (8,  13, 'alpha'),
    (13, 30, 'beta'),
    (30, 40, 'lo-gamma'),
]
DATAGLOVE_FS = 25        # True dataglove sampling rate (stored as ZOH at 1000 Hz)


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
# Step 4a – Band-envelope features  (C, 4, T_new)
# ─────────────────────────────────────────────────────────────────────────────
def compute_band_envelopes(signal: np.ndarray,
                           fs: int = 1000,
                           new_fs: int = DOWNSAMPLE_FS) -> np.ndarray:
    """
    For each band in BAND_ENVELOPES: bandpass → Hilbert envelope → downsample.

    Input:  (C, T) at fs Hz
    Output: (C, n_bands, T_new) at new_fs Hz
    """
    step = fs // new_fs
    C    = signal.shape[0]
    out  = np.zeros((C, len(BAND_ENVELOPES), signal.shape[1] // step),
                    dtype='float32')
    for i, (lo, hi, _) in enumerate(BAND_ENVELOPES):
        b, a     = scipy.signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass')
        filtered = scipy.signal.filtfilt(b, a, signal, axis=1)
        envelope = np.abs(scipy.signal.hilbert(filtered, axis=1))
        out[:, i, :] = envelope[:, ::step].astype('float32')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b – Line-length feature  (C, 1, T_new)
# ─────────────────────────────────────────────────────────────────────────────
def compute_line_length(signal: np.ndarray,
                        fs: int = 1000,
                        new_fs: int = DOWNSAMPLE_FS,
                        window_ms: int = LL_WINDOW_MS) -> np.ndarray:
    """
    Sliding-window line length: sum of |Δsignal| over the last window_ms ms.
    Captures signal complexity / broadband energy.

    Input:  (C, T) at fs Hz
    Output: (C, 1, T_new) at new_fs Hz
    """
    step   = fs // new_fs
    window = int(window_ms * fs / 1000)

    # Cumulative sum of absolute first differences
    diffs  = np.abs(np.diff(signal, axis=1, prepend=signal[:, :1]))  # (C, T)
    cum    = np.cumsum(diffs, axis=1)                                  # (C, T)

    # Causal window: LL[t] = cum[t] - cum[t - window]  (zero-padded for t < window)
    padded = np.concatenate([np.zeros((signal.shape[0], window), 'float32'), cum], axis=1)
    ll     = (cum - padded[:, :cum.shape[1]]).astype('float32')        # (C, T)

    return ll[:, ::step][:, np.newaxis, :]                             # (C, 1, T_new)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4c – Morlet wavelet spectrograms  (C, 40, T_new)
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
# Step 4 (combined) – All features concatenated  →  (C, 45, T_new)
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_features(filtered_signal: np.ndarray,
                         fs: int = 1000,
                         new_fs: int = DOWNSAMPLE_FS) -> np.ndarray:
    """
    Compute and concatenate all features along the frequency axis:
      [band_envelopes (4) | wavelets (40) | line_length (1)] = 45 features/channel

    Input:  (C, T) filtered signal at fs Hz
    Output: (C, 45, T_new) at new_fs Hz
    """
    band_env = compute_band_envelopes(filtered_signal, fs, new_fs)     # (C, 4,  T_new)
    wavelets = downsample_spectrograms(
        compute_spectrograms(filtered_signal, fs), fs, new_fs)          # (C, 40, T_new)
    ll       = compute_line_length(filtered_signal, fs, new_fs)         # (C, 1,  T_new)

    # Clip T_new to shortest (rounding differences between methods)
    T = min(band_env.shape[2], wavelets.shape[2], ll.shape[2])
    return np.concatenate([band_env[:, :, :T],
                           wavelets[:, :, :T],
                           ll[:, :, :T]], axis=1).astype('float32')     # (C, 45, T)


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
