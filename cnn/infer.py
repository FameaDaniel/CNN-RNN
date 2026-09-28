import numpy as np
from tensorflow import keras

(_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

model = keras.models.load_model("models/cnn_fashion_mnist.keras")

print("Evaluating exported model on test dataset...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")