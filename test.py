from typing import Callable, List
import os

import gym
from gym.wrappers import TimeLimit
import compiler_gym
from compiler_gym.envs.llvm import make_benchmark

import numpy as np
from tqdm import tqdm

from policy import create_policy

def get_benchmark_list():
    benchmarks = []
    f = open("cbench-v1.txt","r")
    for line in f:
        benchmarks.append(line.strip())
    f.close()
    return benchmarks

def geom_mean(list):
    list = np.array(list)
    return list.prod()**(1/len(list))

def eval_test(env: gym.Env, policy: Callable[[np.ndarray], int], episodes=100, max_steps=50):
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            steps_left = max_steps - episode_steps
            action = policy(obs, steps_left, episode_steps)
            obs, reward, done, info = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if done:
                pbar.set_postfix(
                    {"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)

def test():
    max_episode_steps = 300
    bench_list = get_benchmark_list()
    rewards = []
    for bench_name in bench_list:
        benchmark_str = f'cbench-v1/{bench_name}'
        env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark_str)#, reward_space="IrInstructionCountNorm")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        policy = create_policy(env, './models/default.model', max_episode_steps)
        return_mean = eval_test(env, policy, episodes=1, max_steps=max_episode_steps)
        rewards.append(return_mean)
    g_mean = geom_mean(rewards)
    print(f'geo mean: {g_mean}')


if __name__ == "__main__":
    test()