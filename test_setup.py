# test_setup.py
import gymnasium as gym
import time

env = gym.make('LunarLander-v3', render_mode="human")
observation, info = env.reset()
print(f"Initial observation: {observation}")

complete = False
episode_reward = 0.0

while not complete:
    action = env.action_space.sample() # random actions
    obs, reward, terminated, truncated, info = env.step(action)
    complete = terminated or truncated
    episode_reward += reward
    time.sleep(1/60)

print("Episode finished, total reward:", episode_reward)
env.close()