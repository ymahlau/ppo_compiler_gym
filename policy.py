import gym
import numpy as np

from policy_critic_network import policy_critic_network
import torch
import pickle

def load_dic(save_dst):
    with open(save_dst, 'rb') as f:
        data = pickle.load(f)
        return data

def create_policy(env: gym.Env, save_dst: str, max_episode_lenght: int):
    # print(
        # f"Creating policy for environment {env} with observation space {env.observation_space} and action space {env.action_space}")
    policy_net = policy_critic_network(env.observation_space.shape[0] + 1, env.action_space.n)
    policy_net.load_state_dict(torch.load(save_dst))
    table_dst = './data/table'
    table = load_dic(table_dst)

    def policy(obs: np.ndarray, steps_left: int, steps_done: int) -> int:
        """
        Receives an observation `obs` and returns the action for the given environment.
        Parameters
        ----------
        obs : np.ndarray
            Observation of the current state of the environment.
        steps_left: Number of steps left in the environment
        steps_done: Number of steps already done in the env
        Returns
        -------
        int
            Action to take in the given state.
        """
        assert env.action_space is not None
        # TODO: replace this random policy
        # return env.action_space.sample()
        obs_concat = np.concatenate((obs, [steps_done]), axis=0)
        obs_tuple = tuple(obs_concat)
        if obs_tuple in table:
            return table[obs_tuple]
        
        # concatenate original observation with number of steps left
        step_tensor = torch.tensor([steps_left])
        obs_tensor = torch.tensor(obs)
        new_obs = torch.cat((obs_tensor, step_tensor), dim=0).float()

        # scores = policy_net.get_action_probs(new_obs)
        # action = torch.argmax(scores, dim=0)
        action, log_prob = policy_net.act(new_obs)
        return action.detach().cpu().item()


    return policy
