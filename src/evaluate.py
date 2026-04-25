from agent import RLAgent
import numpy as np


def evaluate_trained_agent(env_id='CartPole-v1', algo='PPO', timesteps=50000):
    print(f"Training {algo} for {timesteps:,} steps...")
    agent = RLAgent(env_id, algo)
    agent.train(total_timesteps=timesteps)

    metrics = agent.evaluate(n_episodes=20)
    print(f"\nFinal Evaluation (20 episodes):")
    print(f"  Mean Reward: {metrics['mean_reward']}")
    print(f"  Std Reward:  {metrics['std_reward']}")

    if metrics['mean_reward'] >= 400:
        print("  Status: SOLVED (reward >= 400)")
    elif metrics['mean_reward'] >= 200:
        print("  Status: GOOD (reward >= 200)")
    else:
        print("  Status: LEARNING")

    return metrics


if __name__ == '__main__':
    evaluate_trained_agent()