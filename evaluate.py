from typing import Callable, List
import os

import gym
from gym.wrappers import TimeLimit
import compiler_gym
from compiler_gym.envs.llvm import make_benchmark

import numpy as np
from tqdm import tqdm

from policy import create_policy


def evaluate(env: gym.Env, policy: Callable[[np.ndarray], int], episodes=100, max_steps=50):
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


if __name__ == "__main__":
    print(compiler_gym.COMPILER_GYM_ENVS)
    custom_benchmark = make_benchmark(os.path.join(os.path.abspath(
        os.path.dirname(__file__)), "custom_benchmarks", "rot13.cpp"))
    benchmark = "cbench-v1/dijkstra"
    env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark,  reward_space="IrInstructionCountNorm")
    max_episode_steps = 300
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    policy = create_policy(env, './models/default.model', max_episode_steps)
    return_mean = evaluate(env, policy, max_steps=max_episode_steps)
    print(return_mean)