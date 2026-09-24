import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Include parent directory to import shared functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rnn import sanitize_data, create_sequences, FEATURES, TARGET_COL, WINDOW_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH


def evaluate_naive_baseline(window_size: int = WINDOW_SIZE):
    train_raw = pd.read_csv(TRAIN_DATA_PATH)
    test_raw = pd.read_csv(TEST_DATA_PATH)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = FEATURES.index(TARGET_COL)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    X_test, y_test = create_sequences(test_scaled, target_idx, window_size)

    # Naive Persistence: Predict that meantemp at t+1 is equal to meantemp at t (last day of window)
    y_naive = X_test[:, -1, target_idx]

    naive_mse = np.mean((y_test - y_naive) ** 2)
    naive_mae = np.mean(np.abs(y_test - y_naive))

    target_range = scaler.data_range_[target_idx]
    naive_mae_real = naive_mae * target_range

    print("=" * 60)
    print("NAIVE PERSISTENCE BASELINE EVALUATION (y_{t+1} = y_t)")
    print("=" * 60)
    print(f"Lookback window:       {window_size} days")
    print(f"Evaluated sequences:   {len(y_test)}")
    print(f"Normalized MSE:        {naive_mse:.5f}")
    print(f"Normalized MAE:        {naive_mae:.5f}")
    print(f"Real MAE (in Celsius): {naive_mae_real:.2f} C")
    print("=" * 60)
    print("Any recurrent network must achieve lower MAE than this baseline.")
    print("Skill Score = 1 - (MAE_RNN / MAE_Naive)")
    print("=" * 60)

    return {
        "naive_mse": naive_mse,
        "naive_mae": naive_mae,
        "naive_mae_real": naive_mae_real
    }


if __name__ == "__main__":
    evaluate_naive_baseline()
