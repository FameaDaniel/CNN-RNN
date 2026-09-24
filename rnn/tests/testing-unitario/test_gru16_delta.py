import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import callbacks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rnn import (
    sanitize_data,
    create_sequences,
    build_recurrent_model,
    FEATURES,
    TARGET_COL,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    WINDOW_SIZE
)


def run_gru16_delta_test():
    train_raw = pd.read_csv(TRAIN_DATA_PATH)
    test_raw = pd.read_csv(TEST_DATA_PATH)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = FEATURES.index(TARGET_COL)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    # Sequences with DELTA target
    X_train, y_train_delta = create_sequences(train_scaled, target_idx, WINDOW_SIZE, predict_delta=True)
    X_test, y_test_delta = create_sequences(test_scaled, target_idx, WINDOW_SIZE, predict_delta=True)

    target_range = scaler.data_range_[target_idx]
    target_min = scaler.data_min_[target_idx]

    # Reconstructed temperatures
    last_observed_test_scaled = X_test[:, -1, target_idx]
    y_test_abs_real = (last_observed_test_scaled + y_test_delta) * target_range + target_min
    naive_pred_real = last_observed_test_scaled * target_range + target_min
    naive_mae_real = np.mean(np.abs(y_test_abs_real - naive_pred_real))

    # Set seeds for reproducibility
    tf.keras.utils.set_random_seed(42)

    model = build_recurrent_model(
        cell_type="GRU",
        window_size=WINDOW_SIZE,
        num_features=len(FEATURES),
        units=16,
        learning_rate=0.001
    )

    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("=" * 70)
    print("TESTING GRU-16 WITH DELTA PREDICTION AND ADAPTIVE CALLBACKS")
    print(f"Window: {WINDOW_SIZE} days | Units: 16 | Params: {model.count_params()}")
    print(f"Naive Baseline (No Change, Delta=0) MAE: {naive_mae_real:.2f} C")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train_delta,
        validation_data=(X_test, y_test_delta),
        epochs=100,
        batch_size=32,
        callbacks=cb_list,
        verbose=1
    )

    y_pred_delta_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_abs_real = (last_observed_test_scaled + y_pred_delta_scaled) * target_range + target_min

    model_mae_real = np.mean(np.abs(y_test_abs_real - y_pred_abs_real))
    skill_score = 1.0 - (model_mae_real / naive_mae_real)

    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION - GRU-16 (DELTA TARGET):")
    print(f"  Reconstructed Temp MAE: {model_mae_real:.2f} C")
    print(f"  Naive Baseline MAE:     {naive_mae_real:.2f} C")
    print(f"  Skill Score:            {skill_score * 100:+.2f}%")
    print(f"  Stopped at epoch:       {len(history.history['loss'])}")
    print("=" * 70)

    # Plot results
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "GRU-16"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "delta-01.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Subplot 1: Loss curves with learning rate and early stopping indicator
    axes[0].plot(history.history["loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", lw=2, linestyle="--")
    best_epoch = np.argmin(history.history["val_loss"])
    axes[0].axvline(best_epoch, color="red", linestyle=":", label=f"Best Model (Epoch {best_epoch + 1})")
    axes[0].set_title("GRU-16 Delta Loss Convergence (MSE)\nwith EarlyStopping & ReduceLROnPlateau", fontsize=11)
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss (MSE)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Ground Truth vs Predicted Absolute Temperature
    days = np.arange(len(y_test_abs_real))
    axes[1].plot(days, y_test_abs_real, label="Ground Truth (Actual Temp)", color="black", lw=2)
    axes[1].plot(days, y_pred_abs_real, label="Predicted Temp (via Delta)", color="#2ca02c", lw=2, linestyle="-.")
    axes[1].plot(days, naive_pred_real, label="Naive Persistence (y_t)", color="gray", lw=1, linestyle=":", alpha=0.7)
    axes[1].set_title(
        f"Temperature Prediction via Delta\nGRU-16 MAE: {model_mae_real:.2f} C | Naive MAE: {naive_mae_real:.2f} C | Skill: {skill_score*100:+.1f}%",
        fontsize=11
    )
    axes[1].set_xlabel("Test Days", fontsize=10)
    axes[1].set_ylabel("Mean Temperature (C)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"Plot saved successfully to: {output_file}")
    return model_mae_real, skill_score, output_file


if __name__ == "__main__":
    run_gru16_delta_test()
