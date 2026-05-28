import numpy as np
import tensorflow as tf

from models.kinetic_model1 import KineticModel1

if __name__ == "__main__":
    print("PFR Equation PINN")
    model = KineticModel1(0.06, 0.5, 0.001)

    v = np.arange(0.0, 5.0, 0.001)

    # Tensor representation
    v_t = tf.expand_dims(tf.convert_to_tensor(v, dtype=tf.float32), axis=-1)

    model.train_model(v_t, batch_size=20)
