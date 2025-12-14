import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from dqn_agent import DQNAgent

def train_lunar_lander(episodes=1000, render_every=100, config_name="default"):
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    preload_path = "data/my_lander_dataset.npz"   # <-- set to your file
    try:
        agent.replay_buffer.load_from_file(preload_path)
        print(f"Preloaded replay buffer from {preload_path}. Buffer size: {agent.replay_buffer.size()}")
        # If you want training to start earlier, you can reduce train_start
        # e.g., agent.train_start = min(agent.train_start, agent.replay_buffer.size())
    except Exception as e:
        print(f"No preload dataset loaded (looking for {preload_path}): {e}")


    scores = []
    avg_scores = []
    epsilon_values = []

    print(f"\n{'='*60}")
    print(f"Training DQN Agent Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Episodes: {episodes}\n")

    total_steps = 0
    train_frequency = 4  # only train every 4 environment steps

    # training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # choose action
            action = agent.act(state)

            # take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # remember experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent only every 4 steps
            if total_steps % train_frequency == 0:
                agent.replay()

            total_steps += 1
            state = next_state
            total_reward += reward
            steps += 1

        # update target network every N episodes
        if episode % agent.update_target_every == 0:
            agent.update_target_network()

        # Record metrics
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        epsilon_values.append(agent.epsilon)

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Score: {total_reward:7.2f} | "
                  f"Avg(100): {avg_score:7.2f} | Epsilon: {agent.epsilon:.3f}")

        # Save best model
        if avg_score >= 200 and episode >= 100:
            print(f"\n✓ Environment solved in {episode} episodes!")
            print(f"  Average score: {avg_score:.2f}")
            agent.save(f'models/lunar_lander_{config_name}_solved.weights.h5')
            break

        # Render occasionally (test)
        if episode % render_every == 0 and episode > 0:
            test_agent(agent, config_name, render=True)

    # Save final model
    agent.save(f'models/lunar_lander_{config_name}_final.weights.h5')

    # Save training data
    training_data = {
        'config': config_name,
        'episodes': episode + 1,
        'scores': scores,
        'avg_scores': avg_scores,
        'epsilon_values': epsilon_values,
        'final_avg_score': avg_score
    }

    with open(f'results/training_data_{config_name}.json', 'w') as f:
        json.dump(training_data, f)

    # Plot results
    plot_training_results(scores, avg_scores, config_name)

    env.close()
    return agent, training_data

def test_agent(agent, config_name, episodes=5, render=True):
    """Test trained agent"""
    render_mode = 'human' if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    test_scores = []

    print(f"\n{'='*40}")
    print(f"Testing Agent - {config_name}")
    print(f"{'='*40}")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        test_scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}")

    avg_test_score = np.mean(test_scores)
    print(f"\nAverage Test Score: {avg_test_score:.2f}")

    env.close()
    return test_scores

def plot_training_results(scores, avg_scores, config_name):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, label='Episode Score')
    plt.plot(avg_scores, label='Average (100 episodes)')
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training Progress - {config_name}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Average Score (100 episodes)')
    plt.title(f'Learning Curve - {config_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/training_plot_{config_name}.png', dpi=300)
    print(f"\n✓ Plot saved to results/training_plot_{config_name}.png")

if __name__ == "__main__":
    agent, data = train_lunar_lander(episodes=2000, config_name="default")
    test_agent(agent, "default", episodes=10, render=True)
