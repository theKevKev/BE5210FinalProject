# BE5210 Final Project — ECoG Finger Flex Decoding

## Overview
Predict 5-finger flexion (1000 Hz) from ECoG spectrograms (BCI Competition IV Dataset 4, 3 subjects).

**Best model:** CNN v4 — AutoEncoder1D (1D U-Net, all skip connections, no pretraining)
- Mean val_corr: **0.5382** | Subj1: 0.548, Subj2: 0.436, Subj3: 0.630

---

## Reproducing from scratch

### 1. Prepare data
```bash
python prepare_data.py
```
Reads `raw_data/raw_training_data.mat` and `raw_data/leaderboard_data.mat`.
Outputs per subject to `cleaned_data/subj{N}/`:
- `specs_train.npy` — (C, 40, T) Morlet wavelet spectrograms, scaled
- `y_train.npy` — (T, 5) finger flex labels, scaled
- `specs_lead.npy` — (C, 40, T_lead) leaderboard spectrograms, scaled
- `ecog_scaler.pkl` / `ff_scaler.pkl` — fitted scalers

### 2. Train
```bash
python -u train_cnn.py 2>&1 | tee logs/cnn_v1.log
```
Checkpoints auto-versioned to `checkpoints/subj{N}_cnn_best_v{V}.pt`.
Metadata saved to `metadata/cnn_v{V}.json`.

Key hyperparameters (in `train_cnn.py`):
- `WINDOW = 256`, `BATCH_SIZE = 32`, `FINETUNE_EPOCHS = 30`
- `LR = 8.42e-5`, `TRAIN_STRIDE = 2`, `SEED = 42`
- Loss: 0.5×MSE + 0.5×(1−cosine_sim), fingers [0,1,2,4] (skip ring)
- Val split: first 80% train, last 20% val

### 3. Generate leaderboard predictions
```bash
python predict_leaderboard.py             # latest checkpoint
python predict_leaderboard.py --version 1 # specific version
```
Output: `predictions/leaderboard_cnn_v{N}.mat`

---

## Architecture

`AutoEncoder1D` in `models.py`:
- Input projection: (C×F, T) → 32 channels
- 5 encoder blocks (stride-2 MaxPool) → 32× temporal compression
- 5 decoder blocks (2× linear upsample)
- Skip connections from all 5 encoder levels to matching decoder levels
- Final 1×1 conv head → 5 finger outputs

Default channels: [32, 32, 64, 64, 128, 128], kernels: [7,7,5,5,5]

## Pipeline details

- **Preprocessing:** bandpass 40–300 Hz, notch 60/120/180/240 Hz, Morlet wavelets (40 log-spaced freqs), 10× downsample to 100 Hz, RobustScaler
- **Time delay:** 200 ms neural-to-motor offset applied at training time (`specs[t]` → `ff[t+200ms]`), corrected at prediction by shifting output back 20 samples
- **Post-processing:** Gaussian smooth (σ=6 @ 100 Hz), cubic upsample to 1000 Hz

