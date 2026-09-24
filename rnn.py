import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Hyperparameters and Configuration
TRAIN_DATA_PATH = "data/DailyDelhiClimateTrain.csv"
TEST_DATA_PATH = "data/DailyDelhiClimateTest.csv"
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]
TARGET_COL = "meantemp"

CELL_TYPE = "GRU"  # Options: "RNN", "LSTM", "GRU"
WINDOW_SIZE = 30
UNITS = 16
PREDICT_DELTA = True
DROPOUT = 0.0
RECURRENT_DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 15


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


def create_sequences(data: np.ndarray, target_idx: int, window_size: int, predict_delta: bool = False):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        if predict_delta:
            delta = data[i + window_size, target_idx] - data[i + window_size - 1, target_idx]
            y.append(delta)
        else:
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
    predict_delta: bool = PREDICT_DELTA,
    dropout: float = DROPOUT,
    recurrent_dropout: float = RECURRENT_DROPOUT,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    reduce_lr_patience: int = REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = REDUCE_LR_FACTOR,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = sanitize_data(train_raw)
    test_df = sanitize_data(test_raw)

    target_idx = features.index(target_col)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(train_scaled, target_idx, window_size, predict_delta=predict_delta)
    X_test, y_test = create_sequences(test_scaled, target_idx, window_size, predict_delta=predict_delta)

    target_range = scaler.data_range_[target_idx]
    target_min = scaler.data_min_[target_idx]

    print(f"Running experiment with {cell_type.upper()} cell:")
    print(f"  Target mode: {'DELTA (y_{t+1} - y_t)' if predict_delta else 'ABSOLUTE (y_{t+1})'}")
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

    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0).flatten()

    if predict_delta:
        # Reconstruct absolute predictions: y_pred_abs = y_last_observed + delta_pred
        last_observed_scaled = X_test[:, -1, target_idx]
        y_pred_abs_scaled = last_observed_scaled + y_pred
        y_test_abs_scaled = last_observed_scaled + y_test

        y_pred_real = y_pred_abs_scaled * target_range + target_min
        y_test_real = y_test_abs_scaled * target_range + target_min

        delta_mae_real = np.mean(np.abs(y_test - y_pred)) * target_range
        temp_mae_real = np.mean(np.abs(y_test_real - y_pred_real))

        # Naive baseline assumes delta = 0
        naive_temp_mae = np.mean(np.abs(y_test)) * target_range
        skill_score = 1.0 - (temp_mae_real / naive_temp_mae)

        print(f"\nResults for {cell_type.upper()} (DELTA TARGET):")
        print(f"  Delta MAE:            {delta_mae_real:.2f} C")
        print(f"  Reconstructed Temp MAE: {temp_mae_real:.2f} C")
        print(f"  Naive Baseline MAE:     {naive_temp_mae:.2f} C")
        print(f"  Skill Score:            {skill_score * 100:+.2f}%")
    else:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        temp_mae_real = test_mae * target_range
        print(f"\nResults for {cell_type.upper()}:")
        print(f"  Test Normalized MSE: {test_loss:.5f}")
        print(f"  Test Real MAE:        {temp_mae_real:.2f} C")

    return model, history, scaler


if __name__ == "__main__":
    run_experiment()
