import keras
import tensorflow as tf

from settings import Logger


class KineticModel1(keras.Model):
    """
    Kinetic model using neural network to predict concentrations.

    The first-order reaction in PFR is considered:

    v * Ca0 * dXa / dV = k * Ca0 * (1 - Xa)

    v * dXa / dV = k * (1 - Xa)

    Args:
        fr: volumetric flowrate [l/s]
        k: initial guess of the reaction rate constant [s^-1]
        lr: learning rate for the optimizer
    """

    def __init__(
        self,
        fr: float,
        k: float = 0.1,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        # Flowrate
        self.fr = tf.Variable(
            tf.convert_to_tensor(fr, dtype=tf.float32), trainable=False
        )

        # Reaction rate constant (must be added as a weight variable)
        self.k = self.add_weight(
            name="k",
            shape=(),
            initializer=keras.initializers.Constant(k),
            trainable=True,
            dtype=tf.float32,
        )

        # Dense layers to predict concentrations
        self.dense_l1 = keras.layers.Dense(20, activation="tanh")
        self.dense_out = keras.layers.Dense(1, activation="sigmoid")

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Track metrics
        self.loss_metric = keras.metrics.Mean(name="loss")

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
            epoch_loss = 0.0
            num_batches = 0
            for x_batch in dataset:
                with tf.GradientTape() as nn_tape:
                    with tf.GradientTape() as model_tape:
                        model_tape.watch(x_batch)
                        Xa_pred = self(x_batch)

                    # 1) Computing dXa/dV
                    dXa_dV = model_tape.gradient(Xa_pred, x_batch)

                    # 2) Physics residual: dXa/dV
                    residual = self.fr * dXa_dV - self.k * (1.0 - Xa_pred)
                    loss_model = tf.reduce_mean(tf.square(residual))

                    # 3) Boundary loss
                    Xa0_pred_at_0 = self(tf.constant([[0.0]], dtype=tf.float32))
                    loss_boundary = tf.reduce_mean(tf.square(Xa0_pred_at_0))

                    # Total loss
                    loss = loss_model + loss_boundary

                # Calculating the gradients
                gradients = nn_tape.gradient(loss, self.trainable_variables)

                # Applying gradients
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                # Add to total epoch loss
                epoch_loss += float(loss)
                num_batches += 1
            Logger.log(
                "info",
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.10f}, k: {self.k.numpy():.6f}",
            )
