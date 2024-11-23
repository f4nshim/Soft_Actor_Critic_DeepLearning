import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import ActorNetwork, QNetwork, ValueNetwork
import numpy as np


class SAC(object):
    def __init__(self, state_dim, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu") 
        self.alpha = args.alpha
        
        self.action_range = [action_space.low, action_space.high]
        
        self.target_update_interval = args.target_update_interval # default=1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        self.critic = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.value = ValueNetwork(state_dim, args.hidden_size).to(device=self.device)
        self.Tvalue = ValueNetwork(state_dim, args.hidden_size).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=args.lr)
        self.update_value_parameters(tau=1) # hard target update, just copy the value of target value network to value network

        self.policy = ActorNetwork(state_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.reg_lambda = 1e-3

    def update_value_parameters(self, tau=1): # hard updates by default
        with torch.no_grad():
            for target_param, source_param in zip(self.Tvalue.parameters(), self.value.parameters()):
                target_param.copy_(target_param * (1.0 - tau) + source_param * tau)

    def select_action(self, state, eval=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]

        # map action from [-1, 1] to [action_space.low, action_space.high]
        scale = self.action_range[1] - self.action_range[0]
        center = (self.action_range[1] + self.action_range[0]) / 2.0
        action = action * scale / 2.0 + center

        return action

    def learn(self, memory, batch_size, updates):
        # Sample a batch from memory
        states_replay, actions_replay, rewards_replay, next_states_replay, masks_replay = memory.sample(batch_size=batch_size)

        states_replay = torch.tensor(states_replay, dtype=torch.float).to(self.device)
        next_states_replay = torch.tensor(next_states_replay, dtype=torch.float).to(self.device)
        actions_replay = torch.tensor(actions_replay, dtype=torch.float).to(self.device)
        rewards_replay = torch.tensor(rewards_replay, dtype=torch.float).to(self.device).unsqueeze(1)
        masks_replay = torch.tensor(masks_replay, dtype=torch.float).to(self.device).unsqueeze(1)

        with torch.no_grad(): # Equation 8
            v_target_next = self.Tvalue(next_states_replay)
            q_value_next = rewards_replay + masks_replay * self.gamma * (v_target_next)
            # mask set q_value_next[done] = 0

        # update soft Q-function
        # Equation 7
        q1, q2 = self.critic(states_replay, actions_replay)  # Two Q-functions to mitigate positive bias in the policy improvement step
        q1_loss = F.mse_loss(q1, q_value_next) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_value_next) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Update policy
        actions, log_probs, mean, log_std = self.policy.sample(states_replay) # Equation 6：the actions are sampled according to the current policy, instead of the replay buffer

        q1_actions, q2_actions = self.critic(states_replay, actions)
        min_q_actions = torch.min(q1_actions, q2_actions) # to mitigate positive bias; We then use the minimum of the Q-functions for the value gradient in Equation 6 and policy gradient in Equation 13

        policy_loss = ((self.alpha * log_probs) - min_q_actions).mean() # Equation 12：Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        reg_loss = self.reg_lambda * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update soft value function
        v = self.value(states_replay)
        
        with torch.no_grad():
            v_target = min_q_actions - (self.alpha * log_probs)

        v_loss = F.mse_loss(v, v_target) # Equation 5：JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        self.value_optim.zero_grad()
        v_loss.backward()
        self.value_optim.step()

        if updates % self.target_update_interval == 0:
            self.update_value_parameters(tau=self.tau)

        return v_loss.item(), q1_loss.item(), q2_loss.item(), policy_loss.item()

