# imports
import torch.nn as nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import random
device = torch.device("cpu")

# We are going to define an Actor and a Critic Net.
# The actor will learn to approximate the value.
# The critic will try to satisfy the actor with policy gradients.


class ActorNet(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128):
        super(ActorNet, self).__init__()

        self.fc_shared = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, num_actions)
        self.fc_log_std = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, state):
        """Forward method implementation."""
        x = self.relu(self.fc_shared(state))

        mu = self.tanh(self.fc_mu(x))  # Note the action range is (-1, 1)
        log_std = self.fc_log_std(x)# Note you assume you learn the log of the standard deviation. So you want to run fc_log_std forward, and then apply softplus.
        log_std = self.softplus(log_std)
        std = torch.exp(log_std)

        gaussian_policy = Normal(mu, std)
        action = gaussian_policy.sample()

        return action, gaussian_policy

# The critic estimates the value, similar to a Q-network.
class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        """Initialize."""
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size) # you want a one layer network with hidden_size.
        self.out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        """Forward method implementation."""
        x = self.relu(self.fc1(state))
        value = self.out(x)
        return value


class GoalMaze:
    def __init__(self, env, corruption = False, eps = 0.01, gamma=0.9, entropy_weighting=1e-2):
        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.entropy_weighting = entropy_weighting
        self.corruption = corruption

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = ActorNet(obs_dim, action_dim).to(device)
        self.critic = CriticNet(obs_dim, hidden_size=64).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, gaussian_policy = self.actor.forward(state) # use the actor.
        log_prob = gaussian_policy.log_prob(action).sum(dim=-1)
        action_np = action.clamp(-1.0, 1.0).cpu().detach().numpy()
        return log_prob, action_np

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        diff = obs[0:2] - self.goal
        if np.linalg.norm(diff) < self.eps:
            reward = 1
            terminated = True
        else:
            reward = 0
        if self.corruption is True:
            if np.linalg.norm(obs[0:2]) < 0.25:
                obs = [obs[0]+random.uniform(-0.2,0.2),obs[1]+random.uniform(-0.2,0.2),obs[2]+random.uniform(-0.1,0.1),obs[3]+random.uniform(-0.1,0.1)]
        return obs, reward, terminated

    def update_model(self, state, log_prob, next_state, reward, done):
        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        predicted_value = self.critic.forward(state) # Use your critic.
        targ_value = reward + self.gamma * mask * self.critic.forward(next_state) # TD target.
        value_loss = F.smooth_l1_loss(predicted_value, targ_value.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (targ_value - predicted_value).detach() # The target is the Q value.
        advantage = advantage.detach()  # not backpropagated
        policy_loss = -log_prob * advantage # This is the advantage policy gradient.
        policy_loss += self.entropy_weighting * -log_prob  # entropy maximization. Improves training.

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

