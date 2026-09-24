import os
import sys
import numpy as np
import pandas as pd
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
    TEST_DATA_PATH,
    WINDOW_SIZE,
    UNITS,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS
)


def run_architecture_benchmark(
    architectures: list[str] = ["RNN", "LSTM", "GRU"],
    window_size: int = WINDOW_SIZE,
    units: int = UNITS,
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE
):
    train_raw = pd.read_csv(TRAIN_DATA_PATH)
    test_raw = pd.read_csv(TEST_DATA_PATH)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = FEATURES.index(TARGET_COL)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    X_train, y_train = create_sequences(train_scaled, target_idx, window_size)
    X_test, y_test = create_sequences(test_scaled, target_idx, window_size)

    # Compute naive baseline for Skill Score computation
    y_naive = X_test[:, -1, target_idx]
    naive_mae = np.mean(np.abs(y_test - y_naive))
    target_range = scaler.data_range_[target_idx]
    naive_mae_real = naive_mae * target_range

    results = []

    print("=" * 75)
    print(f"BENCHMARK: Comparing Architectures (Window: {window_size} days, Units: {units})")
    print(f"Naive Baseline Real MAE: {naive_mae_real:.2f} C (MAE normalized: {naive_mae:.5f})")
    print("=" * 75)

    for cell in architectures:
        # Set seeds for reproducible comparison
        tf.keras.utils.set_random_seed(42)

        model = build_recurrent_model(
            cell_type=cell,
            window_size=window_size,
            num_features=len(FEATURES),
            units=units,
            learning_rate=learning_rate
        )
        param_count = model.count_params()

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        real_mae = test_mae * target_range
        skill_score = 1.0 - (test_mae / naive_mae)

        results.append({
            "Architecture": cell.upper(),
            "Params": param_count,
            "Test MSE": test_loss,
            "Test MAE (norm)": test_mae,
            "Real MAE (C)": real_mae,
            "Skill Score": skill_score
        })
        print(f"[{cell.upper():<5}] Params: {param_count:<5} | Test MAE: {real_mae:.2f} C | Skill Score: {skill_score * 100:.1f}%")

    print("=" * 75)
    df_results = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df_results.to_string(index=False))
    return df_results


if __name__ == "__main__":
    run_architecture_benchmark()
