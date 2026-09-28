import tensorflow as tf
from tensorflow.keras import layers, Sequential

def test_cnn_output_shape():
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
    ])
    
    dummy_input = tf.random.normal((1, 28, 28, 1))
    output = model(dummy_input)
    
    assert output.shape == (1, 14, 14, 32), "Pooling shape mismatch"
    print("Architecture shape tests passed.")

if __name__ == "__main__":
    test_cnn_output_shape()