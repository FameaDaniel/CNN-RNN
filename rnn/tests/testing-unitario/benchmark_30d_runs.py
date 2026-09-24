import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rnn import (
    sanitize_data,
    create_sequences,
    build_recurrent_model,
    FEATURES,
    TARGET_COL,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH
)

WINDOW_SIZE = 30
BATCHES = {
    "tanda-1": 20,
    "tanda-2": 50,
    "tanda-3": 100
}

MODELS_CONFIG = [
    {"name": "SimpleRNN", "cell": "RNN", "units": 32},
    {"name": "LSTM", "cell": "LSTM", "units": 32},
    {"name": "GRU-16", "cell": "GRU", "units": 16},
    {"name": "GRU-8", "cell": "GRU", "units": 8}
]


def run_benchmark():
    train_raw = pd.read_csv(TRAIN_DATA_PATH)
    test_raw = pd.read_csv(TEST_DATA_PATH)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = FEATURES.index(TARGET_COL)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    X_train, y_train = create_sequences(train_scaled, target_idx, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, target_idx, WINDOW_SIZE)

    target_range = scaler.data_range_[target_idx]
    target_min = scaler.data_min_[target_idx]

    # Convert y_test to real Celsius
    y_test_real = y_test * target_range + target_min

    # Naive baseline for 30-day window
    y_naive = X_test[:, -1, target_idx]
    y_naive_real = y_naive * target_range + target_min
    naive_mae_real = np.mean(np.abs(y_test_real - y_naive_real))

    print("=" * 80)
    print(f"BENCHMARK 30-DAY WINDOW | NAIVE BASELINE MAE: {naive_mae_real:.2f} C")
    print(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")
    print("=" * 80)

    summary_records = []

    base_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resultados-unitarios"))
    os.makedirs(base_results_dir, exist_ok=True)

    for tanda_name, epochs in BATCHES.items():
        tanda_dir = os.path.join(base_results_dir, tanda_name)
        os.makedirs(tanda_dir, exist_ok=True)
        print(f"\n>>> Starting {tanda_name.upper()} ({epochs} Epochs) -> Saving to {tanda_dir}")

        for config in MODELS_CONFIG:
            model_name = config["name"]
            cell_type = config["cell"]
            units = config["units"]

            tf.keras.utils.set_random_seed(42)

            model = build_recurrent_model(
                cell_type=cell_type,
                window_size=WINDOW_SIZE,
                num_features=len(FEATURES),
                units=units,
                learning_rate=0.001
            )
            param_count = model.count_params()

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )

            # Predictions
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            y_pred_real = y_pred_scaled * target_range + target_min

            test_mse = np.mean((y_test - y_pred_scaled) ** 2)
            real_mae = np.mean(np.abs(y_test_real - y_pred_real))
            skill_score = 1.0 - (real_mae / naive_mae_real)

            summary_records.append({
                "Tanda": tanda_name,
                "Epochs": epochs,
                "Model": model_name,
                "Units": units,
                "Params": param_count,
                "Test MSE": test_mse,
                "Real MAE (C)": real_mae,
                "Skill Score (%)": skill_score * 100
            })

            print(f"  [{model_name:<9}] Params: {param_count:<5} | Test MAE: {real_mae:.2f} C | Skill: {skill_score*100:+.1f}%")

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Subplot 1: Loss curves
            ax1.plot(history.history["loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
            ax1.plot(history.history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", lw=2, linestyle="--")
            ax1.set_title(f"{model_name} ({units} units) - {epochs} Epochs\nLoss Convergence", fontsize=12)
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("Loss (MSE)", fontsize=10)
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend()

            # Subplot 2: Real vs Predicted
            days = np.arange(len(y_test_real))
            ax2.plot(days, y_test_real, label="Ground Truth", color="black", lw=2)
            ax2.plot(days, y_pred_real, label="Predicted", color="#2ca02c", lw=2, linestyle="-.")
            ax2.set_title(
                f"Test Predictions (MAE: {real_mae:.2f} C | Skill: {skill_score*100:+.1f}%)\nNaive: {naive_mae_real:.2f} C",
                fontsize=12
            )
            ax2.set_xlabel("Test Days", fontsize=10)
            ax2.set_ylabel("Mean Temperature (C)", fontsize=10)
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend()

            plt.tight_layout()
            output_png = os.path.join(tanda_dir, f"{model_name}.png")
            plt.savefig(output_png, dpi=200)
            plt.close(fig)

    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 85)
    print("FINAL BENCHMARK SUMMARY (WINDOW = 30 DAYS)")
    print("=" * 85)
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    run_benchmark()
