import numpy as np
import tensorflow as tf
from dqn_model import DQNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 10  # episodes
        # Warm up the buffer before training
        self.train_start = 1000

        # networks
        self.policy_network = DQNetwork(state_size, action_size, self.learning_rate)
        self.target_network = DQNetwork(state_size, action_size, self.learning_rate)
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_target_network(self):
        self.target_network.model.set_weights(
            self.policy_network.model.get_weights()
        )

    def act(self, state, training=True):
        # epsilon-greedy
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        # use the TF-accelerated forward pass. convert state to tensor.
        state_arr = np.reshape(state, [1, self.state_size]).astype(np.float32)
        state_tensor = tf.convert_to_tensor(state_arr)
        q_vals = self.policy_network.q_values(state_tensor).numpy()  # shape (1, action_size)
        return int(np.argmax(q_vals[0]))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def replay(self):
        # don't start training until we have enough samples
        if self.replay_buffer.size() < max(self.batch_size, self.train_start):
            return 0.0

        # sample batch (numpy arrays)
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # convert to tf tensors once
        states_tf = tf.convert_to_tensor(states.astype(np.float32))
        next_states_tf = tf.convert_to_tensor(next_states.astype(np.float32))

        # batched forward passes (TF-compiled)
        target_q_values = self.policy_network.q_values(states_tf).numpy()    # shape (batch, action_size)
        next_policy_q = self.policy_network.q_values(next_states_tf).numpy() # for Double DQN (optional)
        next_target_q = self.target_network.q_values(next_states_tf).numpy()

        # Build target matrix
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Standard DQN target: reward + gamma * max_a' Q_target(next, a')
                target_q_values[i][actions[i]] = rewards[i] + \
                    self.gamma * np.max(next_target_q[i])

                # If you want Double DQN, replace the line above with:
                # best_next_action = np.argmax(next_policy_q[i])
                # target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_target_q[i][best_next_action]

        # Train the policy network with the prepared targets using TF train_step (fast)
        loss = self.policy_network.train_step(
            tf.convert_to_tensor(states.astype(np.float32)),
            tf.convert_to_tensor(target_q_values.astype(np.float32))
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        try:
            return float(loss.numpy())
        except Exception:
            return 0.0

    def save(self, filepath):
        self.policy_network.save(filepath)

    def load(self, filepath):
        self.policy_network.load(filepath)
        self.update_target_network()
