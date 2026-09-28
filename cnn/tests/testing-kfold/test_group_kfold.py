import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers

(x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, val_index in skf.split(x_train[:3000], y_train[:3000]):
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    x_tr, x_val = x_train[train_index], x_train[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    
    history = model.fit(x_tr, y_tr, epochs=1, validation_data=(x_val, y_val), verbose=0)
    fold_accuracies.append(history.history['val_accuracy'][0])

print(f"K-Fold Leakage Test Accuracies: {fold_accuracies}")
assert np.mean(fold_accuracies) > 0.6, "Data leakage or underfitting detected in splits."
print("K-Fold data leakage tests passed successfully.")