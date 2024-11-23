import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def init_weights(m):
    # source: https://zh-v2.d2l.ai/chapter_deep-learning-computation/parameters.html
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


# Deprecated in SAC-v2
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # the critic evaluates the value of a state and action pair, so we want to incorporate the action right from the very beginning of the input to the neural network
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

        # We then use the minimum of the Q-functions for the value gradient and policy gradient

        self.apply(init_weights)

    def forward(self, state, action):
        merged_input = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.q1_fc1(merged_input))
        x1 = F.relu(self.q1_fc2(x1))
        x1 = self.q1_fc3(x1)

        x2 = F.relu(self.q2_fc1(merged_input))
        x2 = F.relu(self.q2_fc2(x2))
        x2 = self.q2_fc3(x2)

        return x1, x2


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        
        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self.min_log_std = -20
        self.max_log_std = 2
        self.epsilon = 1e-6

        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        # why log_std instead of std?
        # fc_log_std map x into [-R, R], negative number is included in this range.
        # std should not be negative, but log std could be negative
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        
        # Appendix C: Enforcing Action Bounds
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean, log_std
    
