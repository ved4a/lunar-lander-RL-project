import numpy as np
from collections import deque
import random
import os

class ReplayBuffer:    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)
    
    def save_to_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        states = np.array([e[0] for e in self.buffer])
        actions = np.array([e[1] for e in self.buffer])
        rewards = np.array([e[2] for e in self.buffer])
        next_states = np.array([e[3] for e in self.buffer])
        dones = np.array([e[4] for e in self.buffer])
        np.savez_compressed(path, states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
        print(f"Replay buffer saved to {path}")

    def load_from_file(self, path, max_items=None):
        data = np.load(path)
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        next_states = data['next_states']
        dones = data['dones']

        count = len(states)
        if max_items is not None:
            count = min(count, max_items)

        for i in range(count):
            self.add(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]))

        print(f"Loaded {count} transitions from {path} into replay buffer")