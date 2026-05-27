import keras
import tensorflow as tf


class KineticModel1(keras.Model):
    """
    Kinetic model using neural network to predict concentrations.

    Args:
        fr: volumetric flowrate [l/s]
        Ca0: initial concentration of A [mol/l]
        k: reaction rate constant
        lr: learning rate for the optimizer
    """

    def __init__(
        self, fr: float, Ca0: float, k: float = 0.1, lr: float = 0.001
    ) -> None:
        super().__init__()

        # Flowrate
        self.fr = tf.Variable(
            tf.convert_to_tensor(fr, dtype=tf.float32), trainable=False
        )

        # Initial concentration of A
        self.Ca0 = tf.Variable(
            tf.convert_to_tensor(Ca0, dtype=tf.float32), trainable=False
        )

        # Reaction rate constant
        self.k = tf.Variable(tf.convert_to_tensor(k), dtype=tf.float32, trainable=True)

        # Dense layers to predict concentrations
        self.dense_l1 = keras.layers.Dense(20, activation="tanh")
        self.dense_out = keras.layers.Dense(1, activation="linear")

        # Optimizer
        self.optimizer = keras.optimizers.Adadelta(learning_rate=lr)

        # Track metrics
        self.loss_metric = keras.metrics.Mean(name="loss")

    def call(self, x: tf.Tensor, *_, **__) -> tf.Tensor:
        Ca = self.dense_l1(x)
        Ca = self.dense_out(Ca)
        return Ca

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
                        Ca_pred = self(x_batch)

                    # 1) Computing dCa/dV
                    dCa_dV = model_tape.gradient(Ca_pred, x_batch)

                    # 2) Physics residual: dCa/dV
                    residual = self.fr + dCa_dV + self.k * Ca_pred
                    loss_model = tf.reduce_mean(tf.square(residual))

                    # 3) Boundary loss
                    Ca0_pred_at_0 = self(tf.constant([[0.0]], dtype=tf.float32))
                    loss_boundary = tf.reduce_mean(tf.square(Ca0_pred_at_0 - self.Ca0))
                    loss = loss_model + loss_boundary

                # Calculating the gradients
                gradients = nn_tape.gradient(loss, self.trainable_variables)

                # Applying gradients
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                epoch_loss += float(loss)
                num_batches += 1

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.6f}")
