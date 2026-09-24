import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Hyperparameters and Configuration
TRAIN_DATA_PATH = "data/DailyDelhiClimateTrain.csv"
TEST_DATA_PATH = "data/DailyDelhiClimateTest.csv"
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]
TARGET_COL = "meantemp"

WINDOW_SIZE = 14
UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20


def sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.sort_values("date").reset_index(drop=True)

    # Invalidate physical outliers (Delhi atmospheric pressure bounds: 950 to 1050 hPa)
    pressure_outliers = (cleaned["meanpressure"] < 950) | (cleaned["meanpressure"] > 1050)
    cleaned.loc[pressure_outliers, "meanpressure"] = np.nan

    # Interpolate linearly preserving temporal continuity
    cleaned["meanpressure"] = (
        cleaned["meanpressure"].interpolate(method="linear").bfill().ffill()
    )
    return cleaned


def create_sequences(data: np.ndarray, target_idx: int, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def build_rnn_model(window_size: int, num_features: int, units: int, learning_rate: float):
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.SimpleRNN(units=units, activation="tanh"),
        layers.Dense(units=1)
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def run_experiment(
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
    features: list[str] = FEATURES,
    target_col: str = TARGET_COL,
    window_size: int = WINDOW_SIZE,
    units: int = UNITS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS
):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = features.index(target_col)

    # Fit scaler strictly on training split to avoid data leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(train_scaled, target_idx, window_size)
    X_test, y_test = create_sequences(test_scaled, target_idx, window_size)

    print(f"Dataset summary:")
    print(f"  Train sequences: {X_train.shape[0]} | Shape: {X_train.shape}")
    print(f"  Test sequences:  {X_test.shape[0]}  | Shape: {X_test.shape}\n")

    model = build_rnn_model(
        window_size=window_size,
        num_features=len(features),
        units=units,
        learning_rate=learning_rate
    )
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test MSE: {test_loss:.5f} | Test MAE: {test_mae:.5f}")

    return model, history, scaler


if __name__ == "__main__":
    run_experiment()
