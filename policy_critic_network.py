import torch.nn as nn
import torch
from torch.distributions import Categorical


class policy_critic_network(nn.Module):

    def __init__(self, nom_observations, nom_actions):
        super(policy_critic_network, self).__init__()

        self.input_dim = nom_observations
        self.output_dim = nom_actions

        self.features = nn.Sequential(
            nn.Linear(nom_observations, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.action_lin = nn.Linear(128, nom_actions)
        self.v_lin = nn.Linear(128, 1)
        # self.sm = nn.Softmax()

    def forward(self, x):
        features = self.features(x)
        actions = self.action_lin(features)
        actions = torch.nn.functional.softmax(actions, dim=1)
        v = self.v_lin(features)
        return torch.cat([actions, v], dim=1)

    def get_state_value(self, x):
        y = self.features(x)
        v = self.v_lin(y)
        return v

    def evaluate(self, state, action):
        output = self.forward(state)

        action_probs = torch.index_select(output, 1, torch.tensor(range(self.output_dim)))
        dist = Categorical(action_probs)

        logprob_action = dist.log_prob(action)

        state_val = torch.index_select(output, 1, torch.tensor([self.output_dim]))
        dist_entropy = dist.entropy()

        return logprob_action, state_val, dist_entropy

    def get_action_probs(self, x):
        features = self.features(x)
        action_vals = self.action_lin(features)
        action_probs = torch.nn.functional.softmax(action_vals, dim=0)
        return action_probs

    def act(self, x):
        action_probs = self.get_action_probs(x)
        dist = Categorical(action_probs)
        chosen_action = dist.sample()
        logprob_action = dist.log_prob(chosen_action)
        return chosen_action.detach(), logprob_action.detach()
