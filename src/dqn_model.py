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
    
    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model
    
    def predict(self, state):
        return self.model.predict(state, verbose=1)
    
    def train_on_batch(self, states, targets):
        return self.model.fit(states, targets, epochs=1, verbose=1)

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)