import gymnasium as gym
import time

def explore_lunar_lander():
    env = gym.make("LunarLander-v3", render_mode="human")

    print("⋆｡𖦹°‧★ Lunar Lander Environment Exploration")
    print(f"\nObservation Space: {env.observation_space}")
    print(f"    Shape: {env.observation_space.shape}")
    print(f"    High: {env.observation_space.high}")
    print(f"    Low: {env.observation_space.low}")

    print(f"\nAction Space: {env.action_space}")
    print(f"   Actions: 0=Do nothing, 1=Fire left orientation engine, 2=Fire main engine, 3=Fire right orientation engine")

    for episode in range(5):
        observation, info = env.reset()
        episode_reward = 0.0
        step = 0

        print(f"Episode: {episode + 1}:")

        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1

            if terminated or truncated:
                print(f"    Steps: {step}, Reward: {episode_reward:.2f}")
                if episode_reward >= 200:
                    print("!Solved")
                break

            time.sleep(0.02)
    
    env.close()

    print("\n⋆｡𖦹°‧★ Observation Vector Explained:")
    print("   [0] X position")
    print("   [1] Y position")
    print("   [2] X velocity")
    print("   [3] Y velocity")
    print("   [4] Angle")
    print("   [5] Angular velocity")
    print("   [6] Left leg contact (0 or 1)")
    print("   [7] Right leg contact (0 or 1)")

if __name__ == "__main__":
    explore_lunar_lander()