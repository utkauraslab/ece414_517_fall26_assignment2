"""CartPole driver adapted from CMU 15-780 Spring 2025 HW5.

Preserves original defaults and failure penalty; adds Gymnasium, terminal
handling, seeds, CSV logging and frozen greedy evaluation.
"""
import argparse
import csv
from pathlib import Path
import gymnasium as gym
import numpy as np
import q_learning as q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bins', type=int, default=25)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--schedule', choices=['constant', 'linear'], default='linear')
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--decay-episodes', type=int, default=8000)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--output', type=Path, default=Path('results/run.csv'))
    args = p.parse_args()
    if min(args.episodes, args.eval_episodes, args.decay_episodes) < 1 or args.bins < 2:
        p.error('Episode counts must be positive and bins must be >=2.')
    if not (0 <= args.epsilon <= 1 and 0 < args.alpha <= 1 and 0 <= args.gamma <= 1):
        p.error('Invalid epsilon, alpha, or gamma.')
    if args.output.exists() or args.output.with_suffix('.evaluation.csv').exists():
        p.error('Output exists; choose a new path to preserve previous results.')
    np.random.seed(args.seed)
    env = gym.make('CartPole-v1')
    env.action_space.seed(args.seed)
    buckets = q.create_discretization([-4.8, -5, -.418, -5], [4.8, 5, .418, 5], args.bins)
    table = q.create_q_table(args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    try:
        with args.output.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'score', 'training_return', 'epsilon', 'seed', 'bins', 'alpha', 'gamma', 'schedule'])
            for k in range(args.episodes):
                epsilon = args.epsilon if args.schedule == 'constant' else max(.01, 1 - .99*k/args.decay_episodes)
                state, _ = env.reset(seed=args.seed if k == 0 else None)
                ds = q.get_discretized_state(state, buckets)
                score, total = 0, 0.0
                while True:
                    action = q.epsilon_greedy(table, ds, epsilon)
                    ns, reward, terminated, truncated, _ = env.step(action)
                    nds = q.get_discretized_state(ns, buckets)
                    learning_reward = -475.0 if terminated else reward
                    q.q_update(table, ds, action, learning_reward, nds, args.alpha, args.gamma, terminated=terminated)
                    score += 1
                    total += learning_reward
                    ds = nds
                    if terminated or truncated:
                        break
                scores.append(score)
                w.writerow([k+1, score, total, epsilon, args.seed, args.bins, args.alpha, args.gamma, args.schedule])
                if (k+1) % 100 == 0:
                    print(f'Episode {k+1}: last 100 mean score = {np.mean(scores[-100:]):.2f}')
        evaluation = []
        with args.output.with_suffix('.evaluation.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'score', 'seed'])
            for k in range(args.eval_episodes):
                state, _ = env.reset(seed=args.seed + 100000 + k)
                score = 0
                while True:
                    ds = q.get_discretized_state(state, buckets)
                    action = int(np.argmax(table[ds]))
                    state, _, terminated, truncated, _ = env.step(action)
                    score += 1
                    if terminated or truncated:
                        break
                evaluation.append(score)
                w.writerow([k+1, score, args.seed + 100000 + k])
        print(f'Final training mean ({min(1000, len(scores))} episodes): {np.mean(scores[-1000:]):.2f}')
        print(f'Greedy evaluation mean: {np.mean(evaluation):.2f}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
