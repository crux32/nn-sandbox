import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("\n\nTensorflow refreshing\n\n")

    # Custom layers
    x_h = np.random.uniform(0, 1, (1000, 10))
    x = tf.convert_to_tensor(x_h, tf.float32)
    l1 = tf.keras.layers.Dense(
        10, tf.keras.activations.selu, True, tf.keras.initializers.glorot_uniform
    )
