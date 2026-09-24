# Deep Learning Project: CNN & RNN

## CNN

<!-- CNN module documentation will be added here by teammate -->


---

## RNN (Recurrent Neural Network)

### 1. Problem & Dataset Overview
This module implements a recurrent neural network to solve multivariate sequential temperature forecasting using daily climate measurements in Delhi (`rnn/data/DailyDelhiClimateTrain.csv` and `rnn/data/DailyDelhiClimateTest.csv`).

- **Features (4):** `meantemp`, `humidity`, `wind_speed`, `meanpressure`
- **Target:** Daily temperature variation rate ($\Delta y_{t+1} = y_{t+1} - y_t$) to predict next-day mean temperature ($\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}$).
- **Data Quality:** Sensed pressure outliers outside physical atmospheric bounds [950, 1050 hPa] are sanitized via linear temporal interpolation, strictly preserving the $\Delta t = 1 \text{ day}$ sampling rate.
- **Data Split:** Chronological time split (Train: 2013-2016, Test: 2017) with scalers fitted strictly on training data to prevent look-ahead bias and data leakage.

### 2. Selected Architecture: GRU-16
Following comparative benchmarks across 12 experimental runs (SimpleRNN, LSTM, GRU-16, GRU-8 across 20, 50, and 100 epochs), **GRU with 16 units** demonstrated optimal bias-variance tradeoff:
- **Gating Mechanism:** Reset ($r_t$) and Update ($z_t$) gates mitigate the vanishing gradient problem in Backpropagation Through Time (BPTT) via an additive memory pathway.
- **Parameter Efficiency:** 1,073 parameters (compared to 4,769 in LSTM-32 and 1,217 in SimpleRNN-32), avoiding small-dataset overfitting.
- **Optimization:** Adam optimizer with `ReduceLROnPlateau` (factor=0.5, patience=5) and `EarlyStopping` (patience=15, restoring best weights).
- **Benchmark Performance:** Reconstructed Test MAE of **1.22 °C**, successfully outperforming the Naive Persistence Baseline ($1.23 \text{ °C}$) with a positive Skill Score of **+0.52%**.

### 3. Project Structure
```
rnn/
├── data/
│   ├── DailyDelhiClimateTrain.csv   # Historical training set (2013-2016)
│   ├── DailyDelhiClimateTest.csv    # Historical test set (2017)
│   └── sample_30days.csv            # 30-day inference sample
├── models/
│   ├── export_gru16.py              # Model training and artifact export script
│   ├── gru16_delhi.keras            # Frozen trained GRU-16 model archive
│   └── scaler_delhi.joblib          # Fitted MinMaxScaler artifact
├── tests/
│   ├── MAE-ingenuo/                 # Naive persistence baseline benchmark
│   ├── testing-unitario/            # Architectural comparisons & GRU-16 delta test
│   └── resultados-unitarios/        # Loss curves & prediction plots (Tandas 1, 2, 3)
├── infer.py                         # CLI inference runner
└── rnn.py                           # Core model definitions and pipeline
```

### 4. Running Inference (CLI)
To run inference with the trained GRU-16 model:

```bash
# Default execution (uses rnn/data/sample_30days.csv and exported weights)
python rnn/infer.py

# Custom data input
python rnn/infer.py --data rnn/data/DailyDelhiClimateTest.csv
```

### 5. Running Tests & Benchmarks
```bash
# Evaluate naive baseline threshold
python rnn/tests/MAE-ingenuo/test_naive_mae.py

# Run architecture comparison
python rnn/tests/testing-unitario/test_architectures.py

# Run GRU-16 delta experiment
python rnn/tests/testing-unitario/test_gru16_delta.py
```
