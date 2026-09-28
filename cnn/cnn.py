import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import os

os.makedirs("models", exist_ok=True)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full = np.expand_dims(x_train_full, -1)
x_test = np.expand_dims(x_test, -1)

def build_cnn():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
histories = []

for train_index, val_index in skf.split(x_train_full, y_train_full):
    x_train, x_val = x_train_full[train_index], x_train_full[val_index]
    y_train, y_val = y_train_full[train_index], y_train_full[val_index]
    
    model = build_cnn()
    print(f"Training fold {fold_no}...")
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val), verbose=1)
    histories.append(history.history)
    fold_no += 1

model.save("models/cnn_fashion_mnist.keras")
print("Model exported successfully.")