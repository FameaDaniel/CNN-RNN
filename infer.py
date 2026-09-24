import argparse
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from rnn import sanitize_data, FEATURES, TARGET_COL, WINDOW_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for Delhi Climate GRU-16 Recurrent Network")
    parser.add_argument(
        "--data",
        type=str,
        default="data/DailyDelhiClimateTest.csv",
        help="Path to CSV file containing historical climate data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gru16_delhi.keras",
        help="Path to trained .keras model archive."
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="models/scaler_delhi.joblib",
        help="Path to fitted scaler joblib file."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of days to forecast into the future (default: 1)."
    )
    return parser.parse_args()


def run_inference(data_path: str, model_path: str, scaler_path: str, steps: int = 1):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
    if not os.path.isabs(scaler_path):
        scaler_path = os.path.join(base_dir, scaler_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load artifacts
    print(f"Loading model:  {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load and sanitize data
    raw_df = pd.read_csv(data_path)
    clean_df = sanitize_data(raw_df)

    if len(clean_df) < WINDOW_SIZE:
        raise ValueError(f"Input data must contain at least {WINDOW_SIZE} days. Found: {len(clean_df)}")

    target_idx = FEATURES.index(TARGET_COL)
    target_range = scaler.data_range_[target_idx]
    target_min = scaler.data_min_[target_idx]

    # Extract the most recent window of WINDOW_SIZE days
    recent_history = clean_df.tail(WINDOW_SIZE).copy()
    last_date = recent_history["date"].iloc[-1]
    last_temp = recent_history[TARGET_COL].iloc[-1]

    scaled_window = scaler.transform(recent_history[FEATURES])
    current_input = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))

    print("\n" + "=" * 60)
    print("DELHI CLIMATE INFERENCE ENGINE (GRU-16)")
    print("=" * 60)
    print(f"Latest observed date:        {last_date.strftime('%Y-%m-%d')}")
    print(f"Latest observed temperature: {last_temp:.2f} C")
    print(f"Input window size:           {WINDOW_SIZE} days")
    print(f"Forecast horizon:            {steps} day(s)")
    print("=" * 60)

    # Perform prediction
    predicted_delta_scaled = model.predict(current_input, verbose=0).flatten()[0]
    predicted_delta_real = predicted_delta_scaled * target_range
    predicted_temp_real = last_temp + predicted_delta_real

    forecast_date = last_date + pd.Timedelta(days=1)
    print(f"\nFORECAST RESULT FOR NEXT DAY ({forecast_date.strftime('%Y-%m-%d')}):")
    print(f"  Predicted Delta:      {predicted_delta_real:+.2f} C")
    print(f"  Predicted Mean Temp:  {predicted_temp_real:.2f} C")
    print("=" * 60)

    return predicted_temp_real


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        data_path=args.data,
        model_path=args.model,
        scaler_path=args.scaler,
        steps=args.steps
    )
