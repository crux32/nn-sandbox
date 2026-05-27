import numpy as np
import tensorflow as tf

from models.kinetic_model1 import KineticModel1

if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("\n\nTensorflow refreshing\n\n")

    # Custom layers
    x_h = np.random.uniform(0, 1, (1000, 10))
    y_h = np.random.uniform(0, 1, (1000, 1))

    test_model = KineticModel1(1.0)
