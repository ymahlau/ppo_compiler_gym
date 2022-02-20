import pickle
from typing import List, Callable
from torch.serialization import save
from tqdm import tqdm
from policy_critic_network import policy_critic_network
import gym
from gym.wrappers import TimeLimit
import compiler_gym
from compiler_gym.envs.llvm import make_benchmark
import numpy as np
import random
import torch

def geom_mean(list):
    list = np.array(list)
    return list.prod()**(1/len(list))

def create_policy(env: gym.Env, save_dst: str):
    policy_net = policy_critic_network(env.observation_space.shape[0] + 1, env.action_space.n)
    policy_net.load_state_dict(torch.load(save_dst))
    exploration_rate = 0.1

    def policy(obs: np.ndarray, steps_left: int) -> int:
        """
        Receives an observation `obs` and returns the action for the given environment.
        Parameters
        ----------
        obs : np.ndarray
            Observation of the current state of the environment.
        steps_left: Number of steps left in the environment
        Returns
        -------
        int
            Action to take in the given state.
        """
        # assert env.action_space is not None        
        if np.random.uniform(0, 1) < exploration_rate:
            return env.action_space.sample()
        
        # concatenate original observation with number of steps left
        step_tensor = torch.tensor([steps_left])
        obs_tensor = torch.tensor(obs)
        new_obs = torch.cat((obs_tensor, step_tensor), dim=0).float()

        # scores = policy_net.get_action_probs(new_obs)
        # action = torch.argmax(scores, dim=0)
        action, log_prob = policy_net.act(new_obs)
        
        return action.detach().cpu().item()
    return policy


def gather_data_for_env(env, max_steps=50, episodes=100, best_reward=0):
    policy_dst = './models/default.model'
    policy = create_policy(env, policy_dst)
    
    best_obs_seq = []
    best_action_seq = []
    best_reward = best_reward
    best_episode_length = 0

    # temporarily save best result here to not lose data on accident
    temp_save_dst = './data/temp_tuple'


    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs = env.reset()
        
        current_reward = 0
        current_action_seq = []
        current_obs_seq = [obs]
        episode_steps = 0

        done = False
        save_sequences = False
        while not done:
            steps_left = max_steps - episode_steps
            action = policy(obs, steps_left)
            obs, reward, done, info = env.step(action)

            current_reward += reward
            current_action_seq.append(action)
            current_obs_seq.append(obs)
            episode_steps += 1

            if current_reward > best_reward:
                best_reward = current_reward
                best_episode_length = episode_steps
                save_sequences = True  # save complete sequences at end of episode

            if done:
                if save_sequences:
                    best_action_seq = current_action_seq
                    best_obs_seq = current_obs_seq

                pbar.set_postfix(
                    {"episode reward": current_reward, "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return best_obs_seq, best_action_seq, best_episode_length, best_reward


def save_dic(data, save_dst):
    with open(save_dst, 'wb') as f:
        pickle.dump(data, f)

def get_benchmark_list():
    benchmarks = []
    f = open("cbench-v1.txt","r")
    for line in f:
        benchmarks.append(line.strip())
    f.close()
    return benchmarks

""" dictionary has env name as key and value is tuple of:
1. List of observations on path taken (lenght n+1)
2. List of actions taken (length n)
3. time limit
4. reward
"""
def init_dic():
    benchmarks = get_benchmark_list()
    res_dict = dict()
    default_entry = ([], [], 0, 0)
    for b in benchmarks:
        res_dict[b] = default_entry
    return res_dict

def load_dic(save_dst):
    with open(save_dst, 'rb') as f:
        data = pickle.load(f)
        return data


def improve_dic_infinitely(save_dst):
    data_dict = load_dic(save_dst)
    num_episodes = 1000
    max_episode_steps = 100
    benchmark_list = get_benchmark_list()
    benchmark_list.remove('ghostscript')
    while True:
        # init benchmark
        # bench_name = random.choice(benchmark_list)
        bench_name = 'dijkstra'
        
        # for bench_name in benchmark_list:
        print(f'starting {bench_name}')
        benchmark_str = f'cbench-v1/{bench_name}'
        env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark_str,
                        reward_space="IrInstructionCountNorm")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        res_tuple = gather_data_for_env(env, max_steps=max_episode_steps, episodes=num_episodes)
        
        prev_best_result = data_dict[bench_name][3]
        cur_best_result = res_tuple[3]
        if cur_best_result > prev_best_result:
            data_dict[bench_name] = res_tuple
            save_dic(data_dict, save_dst)
            print(f'found new best for {bench_name}: {prev_best_result} -> {cur_best_result}')


def parse_lookup_table(dict_path, table_path, max_episode_steps):
    table = dict()
    data_dict = load_dic(dict_path)
    max_time_limit = 0
    rewards = []
    num_errors = 0

    for bench_name, res_tuple in data_dict.items():
        print(f'started saving {bench_name}')
        obs_list, action_list, time_limit, expected_reward = res_tuple

        # find max time limit
        if time_limit > max_time_limit:
            max_time_limit = time_limit
        rewards.append(expected_reward)

        benchmark_str = f'cbench-v1/{bench_name}'
        env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark_str,
                        reward_space="IrInstructionCountNorm")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.reset()
        actual_reward = 0
        for i in range(max_episode_steps):
            if i < time_limit:
                obs = obs_list[i]
                action = action_list[i]
            else:
                obs = new_obs
                action = 24 # dce

            obs_concat = np.concatenate((obs, [i]), axis=0)
            obs_tuple = tuple(obs_concat)
                
            if obs_tuple in table:
                old_action = table[obs_tuple]
                print(f'error, obs occurs multiple time (step {i}): {obs}, old action: {old_action}, new action: {action}')
                num_errors += 1
            
            table[obs_tuple] = action

            new_obs, reward, done, info = env.step(action)
            actual_reward += reward
        
        print(f'finished {bench_name}, expected_reward: {expected_reward}, actual: {actual_reward}')
        env.close()

    
    print(f'number of errors: {num_errors}')
    print(f'Maximum time limit over all benchmarks: {max_time_limit}')
    g_mean = geom_mean(rewards)
    print(f'geo mean: {g_mean}')
    # save final table
    save_dic(table, table_path)

    return table



def main():
    dict_dst = './data/data_dict'
    # table_dst = './data/table'
    # improve_dic_infinitely(dict_dst)

    
    # data_dict = load_dic(dict_dst)
    # print(data_dict)

    # table = parse_lookup_table(dict_path=dict_dst, table_path=table_dst, max_episode_steps=300)
    # print(table)

    # bench_name = 'dijkstra'
    # benchmark_str = f'cbench-v1/{bench_name}'
    #env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark_str,
    #               reward_space="IrInstructionCountNorm")
    #max_episode_steps = 200
    #env = TimeLimit(env, max_episode_steps=max_episode_steps)

    #res_tuple = gather_data_for_env(env, max_steps=max_episode_steps, episodes=100)
    #print(res_tuple)


if __name__ == '__main__':
    main()


