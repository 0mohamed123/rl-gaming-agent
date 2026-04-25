# RL Gaming Agent

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange)
![Tests](https://img.shields.io/badge/Tests-11%20passing-green)
![Score](https://img.shields.io/badge/CartPole-500%2F500-brightgreen)

Deep Reinforcement Learning agent trained to master CartPole-v1.
Compares PPO, A2C, and DQN algorithms — achieving perfect 500/500 score.
Framework designed to extend to complex games like Resident Evil 4.

## Algorithm Comparison Results

    Environment: CartPole-v1
    Random Agent: 11/500

    PPO  -> Mean Reward: 500.0 +/- 0.0  | Time: 27.3s  SOLVED
    A2C  -> Mean Reward: 500.0 +/- 0.0  | Time: 16.4s  SOLVED
    DQN  -> Mean Reward:  12.0 +/- 1.26 | Time: 10.6s  (needs more timesteps)

    Best Algorithm: PPO / A2C (Perfect Score)

## Quick Start

    git clone https://github.com/0mohamed123/rl-gaming-agent.git
    cd rl-gaming-agent
    pip install stable-baselines3 gymnasium pygame

    # Run comparison
    cd src
    python train.py

    # Run tests
    cd ../tests
    python -m pytest test_agent.py -v

## Usage

    from agent import RLAgent

    agent = RLAgent('CartPole-v1', 'PPO')
    agent.train(total_timesteps=25000)

    metrics = agent.evaluate(n_episodes=10)
    print(f"Mean Reward: {metrics['mean_reward']}")

## Architecture

    Environment (Gymnasium)
        |
    RLAgent (PPO / A2C / DQN)
        |-- MlpPolicy (Neural Network)
        |-- Training Loop
        |-- Evaluation
        |-- Save/Load

## Road to RE4

    Phase 1 (Done): CartPole mastery - perfect 500/500 score
    Phase 2 (Next): Screen capture + input simulation for RE4
    Phase 3 (Next): Custom reward function based on game events
    Phase 4 (Next): Train PPO agent on RE4 environment

## Test Results

    11 passed | 0 failed

    Tests cover: environment creation, env info, random episode,
    agent build, agent info, invalid algorithm, PPO training,
    evaluation metrics, prediction, error handling, A2C training

## Technologies

- Python 3.12
- Stable-Baselines3 (PPO, A2C, DQN)
- Gymnasium (CartPole-v1)
- PyGame
- pytest (11 tests)