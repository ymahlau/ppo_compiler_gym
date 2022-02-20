from numpy import float32, float64
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
from policy_critic_network import policy_critic_network
from llvm_wrapper import llvm_wrapper

#Credit to https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py. Parts of the rollout_buffer and the update method are taken from here.

class rollout_buffer:

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def add_step_data(self,state,action,logprob,reward,is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.is_terminals.append(is_terminal)

class evaluation:

    def geom_mean(list):
        list = np.array(list)
        return list.prod()**(1/len(list))

    def evaluate(benchmarks,model_name, print_progress = True, additional_steps_for_max = 0, max_trials_per_benchmark = 10, max_time_per_benchmark = 10*1):

        if print_progress:
            print("Evaluating {0}:".format(model_name) )

        episode_length = 200

        performances = []

        for benchmark in benchmarks:
            env = llvm_wrapper([benchmark],max_episode_steps=episode_length,steps_in_observation=False)
            long_env = llvm_wrapper([benchmark],max_episode_steps=episode_length + additional_steps_for_max, steps_in_observation=False)
            model =  policy_critic_network(env.observation_space.shape[0],env.action_space.n)
            model.load_state_dict(torch.load("models/{0}.model".format(model_name)))

            max_reward = 0
            best_action_sequence =  []
            total_reward = 0
            trials = 0
            start = time.time()
            while trials < max_trials_per_benchmark and time.time() - start < max_time_per_benchmark:
                trials += 1
                obs = env.reset()
                done = False
                action_sequence = []
                cum_rewards = []
                while not done:
                    action = model.act(torch.tensor(obs).float())[0].item()
                    action_sequence.append(action)
                    obs,reward,done,_ = env.step(action)
                    cum_rewards.append(reward + (cum_rewards[-1] if len(cum_rewards) > 0 else 0))

                if max(cum_rewards) > max_reward:
                    max_reward =  max(cum_rewards)
                    best_action_sequence = action_sequence
                total_reward += max(cum_rewards)
    
            obs = long_env.reset()
            done = False
            cum_of_max = []
            for action in best_action_sequence:
                _,reward,done,_ = long_env.step(action)
                cum_of_max.append(reward + (cum_of_max [-1] if len(cum_of_max ) > 0 else 0))
            while not done:
                action = model.act(torch.tensor(obs).float())[0].item()
                obs,reward,done,_ = long_env.step(action)
                cum_of_max.append(reward + (cum_of_max [-1] if len(cum_of_max ) > 0 else 0))

            if max(cum_of_max) > max_reward:
                print("Improvement! {0} -> {1}".format(max(cum_of_max),max_reward))

            performance = [max(cum_of_max),total_reward / trials, trials]
            performances.append(performance)  

            if print_progress:
                print("Environment: {0}. Found max of {1} and average of {2} in {3} trials.".format(benchmark,performance[0],performance[1], performance[2]))

            env.close()
            long_env.close()

        performances = np.array(performances)

        if print_progress:
            print("Geometric mean of maxima: {0}".format(evaluation.geom_mean(performances[:,0])))
            print("Geometric mean of averages: {0}".format(evaluation.geom_mean(performances[:,1])))

        return evaluation.geom_mean(performances[:,0]),evaluation.geom_mean(performances[:,1])


class PPO:

    def __init__(self,env,name= "default",EPOCHS = 80, eps_clip = 0.2, loss_mse_fac = 0.5, loss_entr_fac = 0.01, learning_rate = 5e-4, trajectories_until_update = 20, mini_batch_size = 100000000):
        self.EPOCHS = EPOCHS
        self.name = name
        self.eps_clip = eps_clip
        self.loss_mse_fac = loss_mse_fac
        self.loss_entr_fac = loss_entr_fac 
        self.learning_rate = learning_rate
        self.trajectories_until_update = trajectories_until_update
        self.mini_batch_size = mini_batch_size
        self.env = env
        self.actor_critic = policy_critic_network(self.env.observation_space.shape[0],self.env.action_space.n)
        self.buffer = rollout_buffer()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()


    def train(self, training_time = None, log_progress = False, progress_log_rate = 30*60, checkpoint_name = None):

        if checkpoint_name is not None:
            self.actor_critic.load_state_dict(torch.load("models/{0}.model".format(checkpoint_name)))
            print("Continuing training on {0}.".format(checkpoint_name))

        start = time.time()
        last_checkpoint = time.time()

        reward_progress = []

        print("Training started.")
        while (training_time is None) or (time.time() - start < training_time):
            self.collect_trajectories(self.trajectories_until_update)
            self.update()

            if log_progress and (time.time() - last_checkpoint > progress_log_rate):
                torch.save(self.actor_critic.state_dict(),"models/{0}.model".format(self.name).format(self.name))
                geo_maxima,geo_averages = evaluation.evaluate(benchmarks,self.name, print_progress = False, additional_steps_for_max = 0, max_trials_per_benchmark = 10, max_time_per_benchmark = 10) #just for tracking progress
                reward_progress.append(geo_averages)
                print("Geo of averages: {0}".format(reward_progress[-1]))
                plt.clf()
                plt.plot(reward_progress)
                plt.savefig("models/{0}.png".format(self.name))
                
                last_checkpoint = time.time()

            self.env.switch_benchmark()

        if log_progress:
            plt.clf()
            plt.plot(reward_progress)
            plt.savefig("models/{0}.png".format(self.name))
            

    def collect_trajectories(self,count):
        for _ in range(count):
            obs = self.env.reset()
            done = False
            while not done:
                obs = torch.tensor(obs).float()
                action,logprob = self.actor_critic.act(obs)
                new_obs, reward, done, info = self.env.step(action.item())
                self.buffer.add_step_data(obs,action,logprob,reward,done)
                obs = new_obs


    def update(self):
        #Calc Advantages
        xpctd_returns = []
        current_xpctd_return = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                current_xpctd_return = 0
            current_xpctd_return = reward + current_xpctd_return
            xpctd_returns.insert(0, current_xpctd_return)
        xpctd_returns = torch.tensor(xpctd_returns)
        xpctd_returns = (xpctd_returns - xpctd_returns.mean()) / (xpctd_returns.std() + 1e-7)

        rollouts_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        rollouts_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        rollouts_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        #Perform update
        for _ in range(self.EPOCHS):
            batch_size = self.mini_batch_size if self.mini_batch_size <=  len(rollouts_states) else len(rollouts_states)

            sampled_indices = torch.tensor(random.sample(range(len(rollouts_states)),batch_size))
            sampled_states = torch.index_select(rollouts_states,0,sampled_indices)
            sampled_actions = torch.index_select(rollouts_actions,0,sampled_indices)
            sampled_logprobs = torch.index_select(rollouts_logprobs,0,sampled_indices)
            sampled_xpctd_returns = torch.index_select(xpctd_returns,0,sampled_indices)

            logprobs, state_values, dist_entropies = self.actor_critic.evaluate(sampled_states, sampled_actions)
            
            state_values = torch.squeeze(state_values)

            prob_ratios = torch.exp( logprobs - sampled_logprobs)

            advantages = (sampled_xpctd_returns - state_values).detach()

            surr1 = prob_ratios * advantages
            surr2 = torch.clamp(prob_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + self.loss_mse_fac*self.mse_loss(state_values, sampled_xpctd_returns) - self.loss_entr_fac*dist_entropies

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer.clear()


benchmarks = []
f = open("cbench-v1.txt","r")
for line in f:
    benchmarks.append(line.strip())
f.close()

env = llvm_wrapper(benchmarks,max_episode_steps=30, steps_in_observation=False)

ppo_training = PPO(env, name = "test", EPOCHS = 8, learning_rate = 2.5e-4, eps_clip= 0.1, trajectories_until_update=100)
ppo_training.train(log_progress=True, training_time = 60*60*1000, progress_log_rate = 60*30)

