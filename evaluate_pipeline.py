"""
evaluate_pipeline.py
====================
Diagnose the data cleaning pipeline without training any model.

Runs three analyses on the already-cleaned data in cleaned_data/:

  1. Feature–label correlation
     For each wavelet frequency band, compute the max Pearson r between
     any single feature and each finger label (at zero lag).  Tells you
     whether the features carry decodable information at all.

  2. Time-delay sweep
     Sweep the neural-to-motor delay from -400 ms to +400 ms and report
     which lag maximises the mean feature–label correlation.  Validates
     (or challenges) our 200 ms assumption.

  3. Label quality
     Reports smoothness (zero-diff fraction), variance per finger, and
     inter-finger correlation.  Near-zero variance or near-all-zero-diffs
     means that finger's label is unusable.

Usage:
  python evaluate_pipeline.py          # all 3 subjects
  python evaluate_pipeline.py --subj 1 # subject 1 only
"""

import argparse
import pathlib
import numpy as np

PROJECT_ROOT   = pathlib.Path(__file__).parent
CLEAN_DATA_DIR = PROJECT_ROOT / 'cleaned_data'

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
DOWNSAMPLE_FS = 100   # Hz — matches prepare_data.py

# Approximate frequency band boundaries (Hz) for labelling output
BAND_EDGES = [40, 70, 100, 150, 200, 300]
BAND_NAMES = ['lo-γ(40-70)', 'mid-γ(70-100)', 'hi-γ(100-150)',
              'vhi-γ(150-200)', 'ultra-γ(200-300)']


def band_of(freq: float) -> str:
    for i, edge in enumerate(BAND_EDGES[1:]):
        if freq <= edge:
            return BAND_NAMES[i]
    return BAND_NAMES[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature–label correlation
# ─────────────────────────────────────────────────────────────────────────────

def feature_label_correlation(specs: np.ndarray, y: np.ndarray,
                               freqs_hz: np.ndarray) -> None:
    """
    For each wavelet frequency, compute the 90th-percentile |r| across all
    channels, then report the best frequency per finger.

    specs : (C, F, T)
    y     : (T, 5)
    """
    C, F, T = specs.shape
    # Flatten to (T, C*F) and z-score for Pearson
    X = specs.reshape(C * F, T).T.astype('float64')
    X -= X.mean(0);  X /= (X.std(0) + 1e-8)
    Y  = y.astype('float64')
    Y -= Y.mean(0);  Y /= (Y.std(0) + 1e-8)

    print('\n  Feature–label correlation (max |r| across channels, per freq band)')
    print(f"  {'Finger':<10}", end='')
    for bn in BAND_NAMES:
        print(f'  {bn:>16}', end='')
    print()

    for fi, fname in enumerate(FINGER_NAMES):
        y_col = Y[:, fi]
        # Per-frequency max |r| across channels
        band_max = {bn: 0.0 for bn in BAND_NAMES}
        for f_idx in range(F):
            bn  = band_of(freqs_hz[f_idx])
            # Correlate all C channels against this finger at this freq
            r_vals = np.abs(
                (X[:, f_idx * C:(f_idx + 1) * C].T @ y_col) / T
            )
            band_max[bn] = max(band_max[bn], float(r_vals.max()))

        print(f'  {fname:<10}', end='')
        for bn in BAND_NAMES:
            print(f'  {band_max[bn]:>16.4f}', end='')
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time-delay sweep
# ─────────────────────────────────────────────────────────────────────────────

def time_delay_sweep(specs: np.ndarray, y: np.ndarray,
                     fs: int = DOWNSAMPLE_FS) -> None:
    """
    Slide the neural window relative to labels by ±400 ms.
    At each lag, compute the mean |r| between the top-10 features and labels.
    Reports the optimal lag per finger and the overall best lag.

    specs : (C, F, T)
    y     : (T, 5)
    """
    C, F, T = specs.shape
    max_lag_ms = 400
    max_lag    = int(max_lag_ms * fs / 1000)   # samples
    lags_ms    = np.arange(-max_lag, max_lag + 1) * (1000 / fs)

    # Use mean over all features as a summary signal (faster than all C*F)
    # Mean over channels for each frequency → (F, T)
    feat_mean = specs.mean(axis=0)              # (F, T)
    feat_mean = (feat_mean - feat_mean.mean(1, keepdims=True))
    feat_mean /= (feat_mean.std(1, keepdims=True) + 1e-8)

    print('\n  Time-delay sweep  (mean |r| between avg-channel features and labels)')
    print(f"  {'Lag (ms)':>10}  " + '  '.join(f'{fn:>8}' for fn in FINGER_NAMES) + '  Mean')

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # spec leads: spec[0:T+lag] vs label[-lag:T]
            s = feat_mean[:, :T + lag]
            l = y[-lag:T]
        elif lag > 0:
            # spec lags: spec[lag:T] vs label[0:T-lag]
            s = feat_mean[:, lag:T]
            l = y[:T - lag]
        else:
            s = feat_mean
            l = y

        T2 = s.shape[1]
        l  = (l - l.mean(0)) / (l.std(0) + 1e-8)
        r_per_finger = np.abs((s @ l) / T2).mean(0)  # (5,)
        results.append((lag, r_per_finger))

    results = np.array([(lag, *r) for lag, r in results])  # (n_lags, 6)
    lags_col = results[:, 0].astype(int)
    r_mat    = results[:, 1:]   # (n_lags, 5)
    mean_r   = r_mat.mean(1)

    best_overall = lags_col[np.argmax(mean_r)]
    best_per_finger = [lags_col[np.argmax(r_mat[:, fi])] for fi in range(5)]

    # Print every 50 ms
    step = int(50 * fs / 1000)
    for i in range(0, len(lags_col), step):
        lag_ms  = lags_col[i] * 1000 / fs
        r_str   = '  '.join(f'{r_mat[i, fi]:8.4f}' for fi in range(5))
        marker  = ' ← best' if lags_col[i] == best_overall else ''
        print(f'  {lag_ms:>10.0f}  {r_str}  {mean_r[i]:.4f}{marker}')

    print(f'\n  Best overall lag : {best_overall * 1000 / fs:+.0f} ms '
          f'(mean r={mean_r[lags_col == best_overall][0]:.4f})')
    print('  Best lag per finger: ' +
          ', '.join(f'{FINGER_NAMES[fi]}={best_per_finger[fi]*1000/fs:+.0f}ms'
                    for fi in range(5)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Label quality
# ─────────────────────────────────────────────────────────────────────────────

def label_quality(y: np.ndarray) -> None:
    """
    y : (T, 5)
    """
    print('\n  Label quality')
    print(f"  {'Finger':<10}  {'mean':>6}  {'std':>6}  {'zero-diff%':>10}  {'max':>6}")
    for fi, fname in enumerate(FINGER_NAMES):
        col = y[:, fi]
        zd  = (np.diff(col) == 0).mean() * 100
        print(f'  {fname:<10}  {col.mean():6.3f}  {col.std():6.3f}  {zd:10.1f}%  {col.max():6.3f}')

    print('\n  Inter-finger Pearson r:')
    header = '  ' + ' '.join(f'{fn[:5]:>6}' for fn in FINGER_NAMES)
    print(header)
    for fi in range(5):
        row = '  ' + FINGER_NAMES[fi][:5].ljust(6)
        for fj in range(5):
            r = float(np.corrcoef(y[:, fi], y[:, fj])[0, 1])
            row += f'{r:6.2f} '
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_subject(subj: int) -> None:
    d = CLEAN_DATA_DIR / f'subj{subj}'
    print(f'\n{"="*60}')
    print(f'Subject {subj}')
    print(f'{"="*60}')

    specs = np.load(d / 'specs_train.npy')   # (C, F, T)
    y     = np.load(d / 'y_train.npy')       # (T, 5)
    C, F, T = specs.shape
    print(f'  specs={specs.shape}  y={y.shape}')

    # Reconstruct approximate wavelet frequencies (must match prepare_data.py)
    from data_processing import L_FREQ, H_FREQ, WAVELET_NUM
    freqs_hz = np.logspace(np.log10(L_FREQ), np.log10(H_FREQ), WAVELET_NUM)

    label_quality(y)
    feature_label_correlation(specs, y, freqs_hz)
    time_delay_sweep(specs, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subj', type=int, default=0,
                   help='Subject number 1-3, or 0 for all (default)')
    args = p.parse_args()

    subjects = [args.subj] if args.subj else [1, 2, 3]
    for s in subjects:
        evaluate_subject(s)


if __name__ == '__main__':
    main()