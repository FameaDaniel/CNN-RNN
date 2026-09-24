import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import callbacks

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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "gru16_delhi.keras")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler_delhi.joblib")


def export_trained_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = sanitize_data(pd.read_csv(TRAIN_DATA_PATH))
    test_df = sanitize_data(pd.read_csv(TEST_DATA_PATH))

    target_idx = FEATURES.index(TARGET_COL)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    X_train, y_train_delta = create_sequences(train_scaled, target_idx, WINDOW_SIZE, predict_delta=True)
    X_test, y_test_delta = create_sequences(test_scaled, target_idx, WINDOW_SIZE, predict_delta=True)

    tf.keras.utils.set_random_seed(42)

    model = build_recurrent_model(
        cell_type="GRU",
        window_size=WINDOW_SIZE,
        num_features=len(FEATURES),
        units=16,
        learning_rate=0.001
    )

    cb_list = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
    ]

    print("Training final GRU-16 model for export...")
    model.fit(
        X_train,
        y_train_delta,
        validation_data=(X_test, y_test_delta),
        epochs=100,
        batch_size=32,
        callbacks=cb_list,
        verbose=0
    )

    # Save serialized Keras model
    model.save(MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")

    # Save fitted scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved scaler to: {SCALER_FILE}")

    # Evaluate exported model
    target_range = scaler.data_range_[target_idx]
    target_min = scaler.data_min_[target_idx]
    y_pred_delta = model.predict(X_test, verbose=0).flatten()

    last_observed_scaled = X_test[:, -1, target_idx]
    y_pred_real = (last_observed_scaled + y_pred_delta) * target_range + target_min
    y_test_real = (last_observed_scaled + y_test_delta) * target_range + target_min

    test_mae = np.mean(np.abs(y_test_real - y_pred_real))
    print(f"Exported model verified with Test MAE: {test_mae:.2f} C")


if __name__ == "__main__":
    export_trained_model()
