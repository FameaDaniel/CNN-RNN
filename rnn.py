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

CELL_TYPE = "GRU"  # Options: "RNN", "LSTM", "GRU"
WINDOW_SIZE = 14
UNITS = 32
DROPOUT = 0.0
RECURRENT_DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20


def sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.sort_values("date").reset_index(drop=True)

    pressure_outliers = (cleaned["meanpressure"] < 950) | (cleaned["meanpressure"] > 1050)
    cleaned.loc[pressure_outliers, "meanpressure"] = np.nan
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


def build_recurrent_model(
    cell_type: str,
    window_size: int,
    num_features: int,
    units: int,
    learning_rate: float,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0
):
    cell_type_upper = cell_type.upper()
    if cell_type_upper == "RNN":
        recurrent_layer = layers.SimpleRNN(
            units=units,
            activation="tanh",
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
    elif cell_type_upper == "LSTM":
        recurrent_layer = layers.LSTM(
            units=units,
            activation="tanh",
            recurrent_activation="sigmoid",
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
    elif cell_type_upper == "GRU":
        recurrent_layer = layers.GRU(
            units=units,
            activation="tanh",
            recurrent_activation="sigmoid",
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
    else:
        raise ValueError(f"Unsupported cell_type: {cell_type}. Choose 'RNN', 'LSTM', or 'GRU'.")

    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        recurrent_layer,
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
    cell_type: str = CELL_TYPE,
    window_size: int = WINDOW_SIZE,
    units: int = UNITS,
    dropout: float = DROPOUT,
    recurrent_dropout: float = RECURRENT_DROPOUT,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS
):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = features.index(target_col)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(train_scaled, target_idx, window_size)
    X_test, y_test = create_sequences(test_scaled, target_idx, window_size)

    print(f"Running experiment with {cell_type.upper()} cell:")
    print(f"  Lookback window: {window_size} days | Hidden units: {units}")
    print(f"  Train sequences: {X_train.shape[0]} | Test sequences: {X_test.shape[0]}\n")

    model = build_recurrent_model(
        cell_type=cell_type,
        window_size=window_size,
        num_features=len(features),
        units=units,
        learning_rate=learning_rate,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
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
    target_range = scaler.data_range_[target_idx]
    real_mae = test_mae * target_range

    print(f"\nResults for {cell_type.upper()}:")
    print(f"  Test Normalized MSE: {test_loss:.5f}")
    print(f"  Test Normalized MAE: {test_mae:.5f}")
    print(f"  Test Real MAE:        {real_mae:.2f} C (range: {target_range:.2f} C)")

    return model, history, scaler


if __name__ == "__main__":
    run_experiment()
