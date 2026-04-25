import sys
sys.path.append('../src')

import numpy as np
import pytest
import gymnasium as gym
from agent import RLAgent
from environment import EnvironmentInfo, make_env


# ===== Environment Tests =====
def test_make_env():
    env = make_env('CartPole-v1')
    assert env is not None
    env.close()


def test_env_info():
    env = make_env('CartPole-v1')
    info = EnvironmentInfo(env)
    data = info.get_info()
    assert data['n_actions'] == 2
    assert data['observation_shape'] == (4,)
    env.close()


def test_random_episode():
    env = make_env('CartPole-v1')
    info = EnvironmentInfo(env)
    result = info.random_episode()
    assert result['total_reward'] > 0
    assert result['steps'] > 0
    env.close()


# ===== Agent Tests =====
def test_agent_build():
    agent = RLAgent('CartPole-v1', 'PPO')
    agent.build()
    assert agent.model is not None


def test_agent_info():
    agent = RLAgent('CartPole-v1', 'PPO')
    info = agent.get_info()
    assert info['algorithm'] == 'PPO'
    assert info['env_id'] == 'CartPole-v1'
    assert info['trained'] == False


def test_agent_invalid_algorithm():
    agent = RLAgent('CartPole-v1', 'INVALID')
    with pytest.raises(ValueError):
        agent.build()


def test_agent_train_ppo():
    agent = RLAgent('CartPole-v1', 'PPO')
    agent.train(total_timesteps=1000)
    assert agent.model is not None
    info = agent.get_info()
    assert info['trained'] == True


def test_agent_evaluate():
    agent = RLAgent('CartPole-v1', 'PPO')
    agent.train(total_timesteps=5000)
    metrics = agent.evaluate(n_episodes=5)
    assert metrics['mean_reward'] > 0
    assert 'std_reward' in metrics
    assert metrics['n_episodes'] == 5


def test_agent_predict():
    agent = RLAgent('CartPole-v1', 'PPO')
    agent.train(total_timesteps=1000)
    env = make_env('CartPole-v1')
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert action in [0, 1]
    env.close()


def test_agent_not_trained_raises():
    agent = RLAgent('CartPole-v1', 'PPO')
    with pytest.raises(RuntimeError):
        agent.evaluate()


def test_a2c_training():
    agent = RLAgent('CartPole-v1', 'A2C')
    agent.train(total_timesteps=1000)
    metrics = agent.evaluate(n_episodes=3)
    assert metrics['mean_reward'] > 0