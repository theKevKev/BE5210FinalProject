# BE5210 Final Project — Submission

Canvas submission files for the algorithm evaluation.

## What to submit

| File | Submit to |
|------|-----------|
| `submission/predict_test.ipynb` | Canvas |
| `submission/algorithm.zip` | Canvas |

## What's in the zip

```
algorithm.zip
├── models.py
├── data_processing.py
├── train_utils.py
├── predict_utils.py
├── checkpoints/
│   ├── subj1_cnn_best_v4.pt
│   ├── subj2_cnn_best_v4.pt
│   └── subj3_cnn_best_v4.pt
└── scalers/
    ├── subj1_ecog_scaler.pkl
    ├── subj2_ecog_scaler.pkl
    └── subj3_ecog_scaler.pkl
```

## How the TAs run it

1. Upload `algorithm.zip` + `truetest_data.mat` to Colab
2. Run all cells in `predict_test.ipynb`
3. Download `predictions.mat`

## Other branches

- `main` — full history, all versions, logs, checkpoints
- `condensed` — minimal reproduction of CNN v4 only
