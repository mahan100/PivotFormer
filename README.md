# PivotFormer: Alternating-Axis Attention for Multivariate Time Series Forecasting

Official implementation of **PivotFormer**, a lightweight channel-independent architecture for long-term multivariate time series forecasting.

## Architecture Highlights

- **PivotAttention (Alternating-Axis Double-Softmax):** Alternates the second softmax normalization axis between temporal patches (`dim=-2`) and latent feature dimensions (`dim=-1`) across successive encoder layers, stabilized by Post-Softmax **RMSNorm** and learnable `output_scale`.
- **Learnable Attention Temperature ($\tau$):** Per-head learnable temperature applied prior to attention softmax to counteract `tanh` logit compression and preserve peak/outlier sensitivity.
- **Gated Residual Forecasting Head:** Combines a `PELU` + `Conv1d` temporal projection with a learnable sigmoid-gated linear skip connection to prevent high-frequency spike smoothing.
- **Combined Objective:** Trained with $\mathcal{L} = 0.8\,\text{MSE} + 0.2\,\text{MAE}$ under **RevIN** normalization.

## Repository Structure

```text
PivotFormer/
├── Train_ETTh1.ipynb          # ETTh1 (Hourly, 7 channels)
├── Train_ETTh2.ipynb          # ETTh2 (Hourly, 7 channels)
├── Train_ETTm1.ipynb          # ETTm1 (15-min, 7 channels)
├── Train_ETTm2.ipynb          # ETTm2 (15-min, 7 channels)
├── Train_Exchange_Rate.ipynb  # Exchange Rate (Daily, 8 channels)
├── Train_Weather.ipynb        # Weather (10-min, 21 channels)
├── results/                   # Lightweight summary CSV/JSON metrics & plots
└── requirements.txt
