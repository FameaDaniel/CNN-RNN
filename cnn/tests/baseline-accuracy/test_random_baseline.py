import numpy as np
from tensorflow import keras

(_, _), (_, y_test) = keras.datasets.fashion_mnist.load_data()

np.random.seed(42)
random_preds = np.random.randint(0, 10, size=y_test.shape)
accuracy = np.mean(random_preds == y_test)

print("--- BASELINE MODEL ---")
print(f"Random Guessing Accuracy: {accuracy:.4f}")