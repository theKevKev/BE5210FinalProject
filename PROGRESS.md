# BE5210 Final Project — Progress Notes

## Project Overview
ECoG → finger flexion decoding (BCI Competition IV Dataset 4, 3 subjects).
Predict 5-finger flexion at 100 Hz from Morlet wavelet spectrograms of ECoG.

---

## Data Pipeline

Run once to build `cleaned_data/`:
```bash
python prepare_data.py
```

Outputs per subject (`cleaned_data/subj{N}/`):
- `specs_train.npy` — (C, 40, T) Morlet wavelet spectrograms, scaled
- `y_train.npy` — (T, 5) finger flex labels, scaled
- `specs_lead.npy` — (C, 40, T_lead) leaderboard spectrograms, scaled

Key parameters (in `data_processing.py`):
- Bandpass: 40–300 Hz, notch at 60/120/180/240 Hz
- Morlet wavelets: 40 log-spaced freqs, 40–300 Hz, downsampled to 100 Hz
- Time delay: 200 ms (specs[t] predicts ff[t+200ms]), corrected at prediction time
- Scaling: RobustScaler(quantile_range=(0.1,0.9)) for ECoG, MinMaxScaler for labels

---

## Model

`AutoEncoder1D` in `models.py`:
- 1D U-Net CNN, 5 encoder + 5 decoder blocks, all stride-2 → 32× temporal compression
- `skip_levels` controls which decoder steps receive skip connections from encoder
  - `None` (default) = all 5 levels active (full U-Net)
  - `{0, 1}` = only two deepest skips active
  - `set()` = pure bottleneck (no skips)
- Default channels: [32, 32, 64, 64, 128, 128], kernels: [7,7,5,5,5]

---

## Training

```bash
# Standard run (recommended — no pretrain, all skips):
python -u train_cnn.py --no-pretrain 2>&1 | tee logs/cnn_vN.log

# With pretraining (not recommended — see experiments below):
python -u train_cnn.py 2>&1 | tee logs/cnn_vN.log

# With selective skip levels:
python -u train_cnn.py --no-pretrain --skip-levels 0 1 2>&1 | tee logs/cnn_vN.log
```

Key hyperparameters (in `train_cnn.py`):
- `WINDOW = 256`, `BATCH_SIZE = 32`, `FINETUNE_EPOCHS = 30`
- `LR = 8.42e-5`, `TRAIN_STRIDE = 2`, `SEED = 42`
- Loss: 0.5×MSE + 0.5×(1−cosine_sim), fingers [0,1,2,4] (skip ring finger)
- Val split: first 80% train, last 20% val
- Checkpoints auto-versioned: `checkpoints/subj{N}_cnn_best_v{V}.pt`
- Metadata saved to `metadata/cnn_v{V}.json`

---

## Predictions

```bash
# Latest checkpoint:
python predict_leaderboard.py --model cnn

# Specific version:
python predict_leaderboard.py --model cnn --version 4
```

Output: `predictions/leaderboard_cnn_v{N}.mat`
- Predictions smoothed (Gaussian σ=6), time-delay corrected (−20 samples), upsampled 100→1000 Hz

---

## Experiment History

### v1 — Pretrain (all skips) + Finetune (all skips)
**Mean val_corr: 0.5235** | Subj1: 0.456, Subj2: 0.470, Subj3: 0.646

Reproduction command (note: no `--no-pretrain` flag, seed=42 set internally):
```bash
python -u train_cnn.py 2>&1 | tee logs/cnn_v1_repro.log
```
Note: pretraining R² was negative (−0.07 to −0.13) — reconstruction was poor even with skips.
Significant overfitting: train_loss ~0.003 vs val_loss ~0.033 by epoch 30.

### v2 — Pretrain (no skips, bottleneck) + Finetune (all skips)
**Failed — negative val_corr.**
Architectural mismatch: encoder trained without skips, fine-tuned with skips. Decoder
randomly initialized and encoder features from bottleneck-only training were incompatible.

### v3 — Pretrain (no skips) + Finetune (encoder frozen 15 epochs, then all skips)
**Mean val_corr: 0.4905** | Subj1: 0.383, Subj2: 0.482, Subj3: 0.606
Frozen encoder warmup fixed v2's mismatch but broken pretrain weights still hurt.

### v4 — No pretrain + Finetune (all skips) ← BEST
**Mean val_corr: 0.5382** | Subj1: 0.548, Subj2: 0.436, Subj3: 0.630

Reproduction command:
```bash
python -u train_cnn.py --no-pretrain 2>&1 | tee logs/cnn_v4_repro.log
```
Key finding: pretraining was slightly hurting performance. Dropping it entirely was better.

### v5 — No pretrain + Finetune (skip levels 0+1 only)
**Mean val_corr: 0.4733** | Subj1: 0.461, Subj2: 0.341, Subj3: 0.618
Pruning skip levels 2–4 hurt more than expected, especially Subject 2.
Individual ablation showed each level contributed <2%, but removing all three together compounded.

---

## Skip Connection Analysis (`analyze_skips.py`)

Run on any trained checkpoint:
```bash
python analyze_skips.py --version 1         # specific version
python analyze_skips.py --subj 1 --version 4  # one subject
```

Key findings from v1 analysis:
- Removing ALL skips: val_corr drops 0.25–0.37 — bottleneck alone is not enough
- Skip levels by impact (0 = deepest encoder output, 4 = shallowest):
  - Level 0: −0.05 to −0.07 (most important)
  - Level 1: −0.04 to −0.08 (important)
  - Level 2: −0.01 to −0.02 (minor)
  - Level 3: ~−0.005 (negligible)
  - Level 4: ~0.000 (none)
- Activation norm ratios: balanced (0.41–0.56) — both paths contribute
- Gradient flow: also balanced; bottleneck path receives slightly more at deeper levels

---

## Open Leads for v6+

1. **Overfitting** — 10–15× train/val loss gap in all versions. Options:
   - Dropout increase
   - Weight decay increase
   - Data augmentation (time shifts, noise injection)
   - More training data (longer windows, more overlap)

2. **Middle finger** — consistently the worst finger across all versions (~0.1–0.4 corr).
   May need subject-specific tuning or a different loss weighting.

3. **Subject 2** — weakest subject, sensitive to architecture changes.
   48 electrodes vs 62/64 for subjects 1/3 — less input signal.

4. **Architecture search** — v4 with all skips is still best. Haven't tried:
   - More epochs (val_corr was still climbing for some subjects at epoch 30)
   - Higher LR warmup + cosine decay
   - Larger model (more channels)
   - Smaller model (fewer channels, addressing overfitting differently)
