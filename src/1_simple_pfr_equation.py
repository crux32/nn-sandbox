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
    v_exp = 0.15  # L/s
    k_true = 0.05  # s^-1
    V = 10.0  # L

    # "Lab" measurements
    num_points = 5
    # Volume points
    v_lab = np.sort(np.random.uniform(0.05, V, size=(num_points, 1)), axis=0).astype(
        np.float32
    )
    # Conversion
    x_lab = analytical_solution_1st_order_equation(k_true, v_exp, v_lab)

    # Volume reactor points
    v = np.arange(0.0, V, 0.01).astype(np.float32)

    # Tensor representation
    v_t = tf.expand_dims(tf.convert_to_tensor(v, dtype=tf.float32), axis=-1)

    # Initialize a model
    model = KineticModel1(v_lab, x_lab, 0.06, 1.0, 0.05)
    model.compile(optimizer=keras.optimizers.Adam())

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=1e-3,  # type: ignore[arg-type]
        patience=4,
        mode="min",
        restore_best_weights=True,
    )

    # Model training
    model.fit(v_t, batch_size=10, epochs=100, shuffle=False, callbacks=[early_stopping])

    # Compare predictions vs "experimental" data
    x_predicted = cast(tf.Tensor, model.predict(v_t))
    x_lab = analytical_solution_1st_order_equation(k_true, v_exp, v)

    # Plot
    fig = plot_true_vs_predicted_PFR(x_lab, tf.reshape(x_predicted, [-1]).numpy(), v)

    fig.show()
