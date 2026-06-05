from typing import cast

import keras
import numpy as np
import tensorflow as tf

from models.kinetic_model1 import KineticModel1
from plotting import plot_true_vs_predicted_PFR

np.random.seed(42)


def analytical_solution_1st_order_equation(
    k: float, fr: float, V: np.ndarray
) -> np.ndarray:
    """Return analytical solution for PFR.

    1st order: X = 1 - exp[-k * V / v]
    """
    return 1 - np.exp(-k * (V / fr))


if __name__ == "__main__":
    print("PFR Equation PINN")

    # Simulated data
    v_exp = 200.0 / 60000.0  # flowrate [L/s] (ml/min to L/s)
    k_true = 0.03  # "True" reaction rate constant [s^-1]
    V = 0.5  # Total volume of the reactor [L]

    # "Lab" measurements
    num_points = 10

    # Volume points
    v_lab = np.sort(np.random.uniform(0.01, V, size=(num_points, 1)), axis=0).astype(
        np.float32
    )
    v_noise = np.random.normal(0, 0.2, v_lab.shape)
    v_lab += v_noise

    # Conversion
    x_lab = analytical_solution_1st_order_equation(k_true, v_exp, v_lab)

    # Volume reactor points
    v = np.arange(0.0, V, 0.001, dtype=np.float32)

    # Tensor representation
    v_t = tf.expand_dims(tf.convert_to_tensor(v, dtype=tf.float32), axis=-1)

    # Initialize a model
    model = KineticModel1(v_lab, x_lab, v_exp, k=0.1)
    model.compile()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=1e-3,  # type: ignore[arg-type]
        patience=4,
        mode="min",
        restore_best_weights=True,
    )

    # Model training
    model.fit(
        v_t, batch_size=10, epochs=1000, shuffle=False, callbacks=[early_stopping]
    )

    # Compare predictions vs "experimental" data
    x_predicted = cast(tf.Tensor, model.predict(v_t))
    x_lab = analytical_solution_1st_order_equation(k_true, v_exp, v)

    # Plot
    fig = plot_true_vs_predicted_PFR(x_lab, tf.reshape(x_predicted, [-1]).numpy(), v)

    fig.show()
