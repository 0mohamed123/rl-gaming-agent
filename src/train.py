from agent import RLAgent
from environment import EnvironmentInfo, make_env
import time


def run_comparison():
    print("=" * 55)
    print("   RL Gaming Agent - Algorithm Comparison")
    print("=" * 55)

    env = make_env('CartPole-v1')
    info = EnvironmentInfo(env)
    env_info = info.get_info()
    print(f"\nEnvironment: {env_info['env_name']}")
    print(f"Observation: {env_info['observation_shape']}")
    print(f"Actions: {env_info['n_actions']}")

    random_result = info.random_episode()
    print(f"\nRandom agent: reward={random_result['total_reward']:.1f} steps={random_result['steps']}")
    env.close()

    algorithms = ['PPO', 'A2C', 'DQN']
    results = {}

    for algo in algorithms:
        print(f"\n--- {algo} ---")
        agent = RLAgent('CartPole-v1', algo)
        start = time.time()
        agent.train(total_timesteps=25000)
        train_time = time.time() - start
        metrics = agent.evaluate(n_episodes=10)
        metrics['train_time'] = round(train_time, 1)
        results[algo] = metrics
        print(f"Mean reward: {metrics['mean_reward']} +/- {metrics['std_reward']}")
        print(f"Training time: {train_time:.1f}s")

    print("\n" + "=" * 55)
    print("Results Summary:")
    for algo, r in results.items():
        print(f"  {algo:5s}: reward={r['mean_reward']:6.1f} | time={r['train_time']}s")

    best = max(results, key=lambda x: results[x]['mean_reward'])
    print(f"\nBest algorithm: {best} (reward={results[best]['mean_reward']})")
    print("=" * 55)
    return results


if __name__ == '__main__':
    run_comparison()