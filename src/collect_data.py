import gymnasium as gym
import numpy as np
import argparse
import os

def collect(env_name="LunarLander-v3", episodes=200, out_path="data/collected_dataset.npz", seed=None):
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        while not done:
            # Choose policy here. Replace env.action_space.sample() with your policy if you have one
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state.astype(np.float32))
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(next_state.astype(np.float32))
            dones.append(bool(done))

            state = next_state
            step += 1

        if (ep + 1) % 10 == 0:
            print(f"Collected episodes: {ep+1}/{episodes}")

    env.close()

    np.savez_compressed(out_path,
                        states=np.array(states),
                        actions=np.array(actions, dtype=np.int32),
                        rewards=np.array(rewards, dtype=np.float32),
                        next_states=np.array(next_states),
                        dones=np.array(dones, dtype=np.bool_))
    print(f"Saved dataset to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='LunarLander-v3')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--out', default='data/collected_dataset.npz')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    collect(env_name=args.env, episodes=args.episodes, out_path=args.out, seed=args.seed)
