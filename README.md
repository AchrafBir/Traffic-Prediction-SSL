# Traffic Prediction with Self-Supervised Learning (SimCLR + GCN-LSTM)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-lightred)
![License](https://img.shields.io/badge/License-MIT-green)

This project demonstrates a self-supervised learning pipeline for traffic forecasting using:
- SimCLR-style contrastive pretraining on traffic time-series windows
- Linear evaluation of the learned representations
- Fine-tuning for multi-step traffic flow prediction

The implementation is provided as a single Jupyter notebook:
`Traffic_Prediction_with_Self_Supervised_Learning.ipynb`.

## Method overview

1. Load datasets (`train.npz`, `val.npz`, `test.npz`)
2. (Optional) Build windowed sequences for forecasting
3. Self-supervised pretraining (SimCLR)
   - Time-series augmentations (jitter, scaling, masking)
   - Encoder: per-feature projection + LSTM temporal encoder + projection head
   - Loss: NT-Xent (contrastive)
4. Linear evaluation
   - Freeze encoder, train a linear regressor on top of representations
5. Fine-tuning
   - Reuse encoder backbone and train a prediction head for traffic forecasting
6. Evaluation and visualization (RMSE/MAE + plots)

## Data format

Place these files next to the notebook:
- `train.npz`
- `val.npz`
- `test.npz`

Each `.npz` must contain:
- `x`: input tensor with shape like `(N, T, num_nodes, num_features)`
- `y`: target tensor with shape compatible with multi-step prediction (commonly `(N, H, num_nodes, flow_types)`)

Notes:
- `num_nodes = x.shape[2]`
- `num_features = x.shape[3]`
- `flow_types` is set to `2` in the notebook.

## Requirements

Python 3.9+ recommended.

Main dependencies (from the notebook imports):
- numpy
- torch
- scikit-learn
- matplotlib
- seaborn (optional, used for plots)

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy torch scikit-learn matplotlib seaborn jupyter
```

## Run

Local:
```bash
jupyter notebook Traffic_Prediction_with_Self_Supervised_Learning.ipynb
```

Colab:
- Upload the notebook and the `.npz` files, or mount Google Drive (cells included in the notebook).

## Outputs (checkpoints)

The notebook saves model weights (local):
- `best_encoder_model3.pth`
- `best_linear_model2.pth`
- `best_finetune_gcn_lstm_model.pth`

It also includes example code to save/load from Google Drive in Colab.

## Important implementation notes

- `adj_mx` is currently created as a random matrix in the notebook. For a real traffic graph, replace it with an adjacency matrix derived from the road network/sensor connectivity.
- The “GCN” block is implemented as a linear layer (feature projection). If you need a true graph convolution, replace it with a graph operator (e.g., spectral/spatial GCN) and use `adj_mx` explicitly in the forward pass.
- Ensure your dataloaders match the training stage:
  - SimCLR pretraining uses pairs of augmented views `(x_i, x_j)`
  - Fine-tuning/evaluation should use `(x, y)` labeled batches

## License

Add your preferred license (e.g., MIT) in `LICENSE`.
