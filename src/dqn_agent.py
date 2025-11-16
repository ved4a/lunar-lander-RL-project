import numpy as np
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
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.policy_network.predict(state)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def replay(self):
        if self.replay_buffer.size() < self.batch_size:
            return 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values
        target_q_values = self.policy_network.predict(states)
        next_q_values = self.target_network.predict(next_states)
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + \
                    self.gamma * np.max(next_q_values[i])
        
        # Train network
        loss = self.policy_network.train_on_batch(states, target_q_values)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.history['loss'][0]
    
    def save(self, filepath):
        self.policy_network.save(filepath)
    
    def load(self, filepath):
        self.policy_network.load(filepath)
        self.update_target_network()