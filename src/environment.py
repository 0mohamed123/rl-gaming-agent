import gymnasium as gym
import numpy as np


def make_env(env_id='CartPole-v1'):
    env = gym.make(env_id)
    return env


def make_atari_env(env_id='ALE/Breakout-v5'):
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    env = gym.make(env_id, render_mode=None)
    env = AtariWrapper(env)
    return env


class EnvironmentInfo:
    def __init__(self, env):
        self.env = env

    def get_info(self):
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        return {
            'env_name': self.env.spec.id if self.env.spec else 'Unknown',
            'observation_space': str(obs_space),
            'observation_shape': obs_space.shape,
            'action_space': str(act_space),
            'n_actions': act_space.n if hasattr(act_space, 'n') else None,
        }

    def random_episode(self):
        obs, _ = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        return {'total_reward': total_reward, 'steps': steps}