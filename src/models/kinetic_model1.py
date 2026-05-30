from typing import cast

import keras
import numpy as np
import tensorflow as tf

from settings import Logger


class KineticModel1(keras.Model):
    """
    Kinetic model using neural network to predict concentrations.

    The first-order reaction in PFR is considered:

    v * Ca0 * dXa / dV = k * Ca0 * (1 - Xa)

    v * dXa / dV = k * (1 - Xa)

    Args:
        x_true: Measured volume points
        y_true Measured conversion
        fr: volumetric flowrate [l/s]
        k: initial guess of the reaction rate constant [s^-1]
        lr: learning rate for the optimizer
        min_delta: Early stopping criteria for convergence
    """

    def __init__(
        self,
        x_true: np.ndarray,
        y_true: np.ndarray,
        fr: float,
        k: float = 0.1,
        lr: float = 0.001,
        min_delta: float = 1e-3,
    ) -> None:
        super().__init__()

        # Known data points
        self.x_true = tf.Variable(
            tf.convert_to_tensor(x_true), dtype=tf.float32, trainable=False
        )
        self.y_true = tf.Variable(
            tf.convert_to_tensor(y_true), dtype=tf.float32, trainable=False
        )

        # For the early stopping criteria
        self.min_delta = min_delta

        # Flowrate
        self.fr = tf.Variable(
            tf.convert_to_tensor(fr, dtype=tf.float32), trainable=False
        )

        # Reaction rate constant (must be added as a weight variable)
        self.k = self.add_weight(
            name="k",
            shape=(),
            # Intentional initialization of the reaction rate constant
            # since it is modified during the training using softplus function
            # to keep the value positive
            initializer=keras.initializers.Constant(np.log(np.exp(k) - 1.0)),
            trainable=True,
            dtype=tf.float32,
        )

        # Dense layers to predict concentrations
        self.dense_l1 = keras.layers.Dense(10, activation="tanh")
        self.dense_out = keras.layers.Dense(1, activation="sigmoid")

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Track metrics
        self.loss_metric = keras.metrics.Mean(name="loss")

        self.__zero_tensor = tf.Variable(
            tf.constant([[0.0]], dtype=tf.float32), trainable=False
        )

    def call(self, x: tf.Tensor, *_, **__) -> tf.Tensor:
        x = self.dense_l1(x)
        x = self.dense_out(x)
        return x

    def train_model(
        self,
        x_data: tf.Tensor,
        batch_size: int = 10,
        epochs: int = 100,
    ):
        """
        Manual training loop for PINN-style physics loss.

        Args:
            x_data: input tensor (collocation points)
            batch_size: batch size
            epochs: number of full passes over data
        """
        dataset = tf.data.Dataset.from_tensor_slices(x_data).batch(batch_size)

        for epoch in range(epochs):
            for x_batch in dataset:
                with tf.GradientTape() as nn_tape:
                    with tf.GradientTape() as model_tape:
                        model_tape.watch(x_batch)
                        Xa_pred = self(x_batch)

                    # 1) Computing dXa/dV
                    dXa_dV = model_tape.gradient(Xa_pred, x_batch)

                    # 2) Physics residual: dXa/dV
                    k_positive = tf.math.softplus(self.k)  # maintain k always positive!
                    residual = self.fr * dXa_dV - k_positive * (1.0 - Xa_pred)
                    loss_model = tf.reduce_mean(tf.square(residual))

                    # 3) Boundary loss
                    Xa0_pred_at_0 = self(tf.constant([[0.0]], dtype=tf.float32))
                    loss_boundary = tf.reduce_mean(tf.square(Xa0_pred_at_0 - 1e-6))

                    # 4) Data loss
                    Xa_pred = self(self.x_true)
                    loss_data = tf.reduce_mean(tf.square(Xa_pred - self.y_true))

                    # Total loss
                    loss = loss_model + loss_boundary + (10 * loss_data)

                    self.loss_metric.update_state(loss)

                # Calculating the gradients
                gradients = nn_tape.gradient(loss, self.trainable_variables)

                # Applying gradients
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            epoch_loss = cast(np.ndarray, self.loss_metric.result().numpy())
            k_physical = tf.math.softplus(self.k).numpy()

            Logger.log(
                "info",
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.10f}, k: {k_physical:.6f}",
            )

            # epoch_loss

            self.loss_metric.reset_state()

    def train_step(self, data):
        x_batch = data
        with tf.GradientTape() as nn_tape:
            with tf.GradientTape() as model_tape:
                model_tape.watch(x_batch)
                Xa_pred = self(x_batch)

            # 1) Computing dXa/dV
            dXa_dV = model_tape.gradient(Xa_pred, x_batch)

            # 2) Physics residual: dXa/dV
            k_positive = tf.math.softplus(self.k)  # maintain k always positive!
            residual = self.fr * dXa_dV - k_positive * (1.0 - Xa_pred)
            loss_model = tf.reduce_mean(tf.square(residual))

            # 3) Boundary loss
            Xa0_pred_at_0 = self(self.__zero_tensor)
            loss_boundary = tf.reduce_mean(tf.square(Xa0_pred_at_0 - 1e-6))

            # 4) Data loss
            Xa_pred = self(self.x_true)
            loss_data = tf.reduce_mean(tf.square(Xa_pred - self.y_true))

            # Total loss
            loss = loss_model + loss_boundary + (10 * loss_data)

            self.loss_metric.update_state(loss)

        # Calculating the gradients
        gradients = nn_tape.gradient(loss, self.trainable_variables)

        # Applying gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": self.loss_metric.result()}

    @property
    def metrics(self):
        return [self.loss_metric]
