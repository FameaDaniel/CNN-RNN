import os
import matplotlib.pyplot as plt
from tensorflow import keras

os.makedirs("fashion_samples", exist_ok=True)
(x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()

for i in range(5):
    plt.imsave(f"fashion_samples/sample_{i}_class_{y_train[i]}.png", x_train[i], cmap='gray')

print("Sample images saved in data/fashion_samples/")