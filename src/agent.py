import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


class RLAgent:
    def __init__(self, env_id='CartPole-v1', algorithm='PPO'):
        self.env_id = env_id
        self.algorithm = algorithm
        self.env = gym.make(env_id)
        self.model = None
        self._algorithms = {
            'PPO': PPO,
            'DQN': DQN,
            'A2C': A2C
        }

    def build(self, **kwargs):
        algo_class = self._algorithms.get(self.algorithm)
        if not algo_class:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        self.model = algo_class(
            'MlpPolicy', self.env,
            verbose=0, **kwargs
        )
        return self

    def train(self, total_timesteps=50000):
        if not self.model:
            self.build()
        print(f"Training {self.algorithm} on {self.env_id}...")
        print(f"Total timesteps: {total_timesteps:,}")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training complete!")
        return self

    def evaluate(self, n_episodes=10):
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env,
            n_eval_episodes=n_episodes, deterministic=True
        )
        return {
            'mean_reward': round(float(mean_reward), 2),
            'std_reward': round(float(std_reward), 2),
            'n_episodes': n_episodes
        }

    def predict(self, obs):
        if not self.model:
            raise RuntimeError("Model not trained.")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path='models/agent'):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path='models/agent'):
        algo_class = self._algorithms.get(self.algorithm)
        self.model = algo_class.load(path, env=self.env)
        return self

    def get_info(self):
        return {
            'algorithm': self.algorithm,
            'env_id': self.env_id,
            'policy': 'MlpPolicy',
            'trained': self.model is not None
        }