import tensorflow as tf
import keras
from keras import layers
import numpy as np

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.model = self._build_model()

        # introduce optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        return model
    
    @tf.function
    def q_values(self, states):
        """
        states: tf.Tensor shaped (batch, state_size) or (1, state_size)
        returns: tf.Tensor shaped (batch, action_size)
        """
        return self.model(states, training=False)

    # Custom train step to avoid train_on_batch overhead and to be TF graph compiled
    @tf.function
    def train_step(self, states, targets):
        """
        states: tf.Tensor (batch, state_size)
        targets: tf.Tensor (batch, action_size)
        returns: scalar loss (tf.float32)
        """
        with tf.GradientTape() as tape:
            preds = self.model(states, training=True)
            loss = tf.reduce_mean(tf.keras.losses.MSE(targets, preds))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    # Small compatibility helpers (still available if some code calls them)
    def predict(self, state):
        # keep for compatibility: returns numpy array
        state = np.asarray(state, dtype=np.float32)
        probs = self.model(state, training=False)
        return probs.numpy()

    def train_on_batch(self, states, targets):
        # compatibility wrapper: convert to tensors and call train_step (returns numpy float)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)
        loss = self.train_step(states, targets)
        return float(loss.numpy())

    def save(self, filepath):
        # save weights only
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)