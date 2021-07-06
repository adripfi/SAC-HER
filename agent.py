import torch
from itertools import chain as concatenate
import numpy as np
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
import copy
from model import Actor, Critic


class Agent:
    def __init__(self, state_dim, action_dim, device, lr, gamma=0.99, tau=0.005, alpha=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # two critics + target networks to mitigate overestimation bias using (clipped) double q trick
        self.q1 = Critic(num_inputs=1)
        self.q2 = Critic(num_inputs=1)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.q1_optimizer = torch.optim.Adam()

        self.policy = Actor()

    def sample(self, state):
        state = torch.as_tensor(state).to(self.device)
        action, _ = self.policy.forward(state)

        return action

    def get_policy_loss(self, states):
        next_actions, pi_log_probs = self.policy(states)
        q1_target = self.q1_target.forward(states, next_actions)
        q2_target = self.q2_target.forward(states, next_actions)
        q_target_min = torch.min(q1_target, q2_target)
        pi_loss = (self.alpha * pi_log_probs - q_target_min).mean()

        return pi_loss

    def get_q_loss(self, states, actions, rewards, dones, next_states):
        # compute TD target using target q networks
        with torch.no_grad():
            # sample target actions from current policy
            next_actions, pi_log_probs = self.policy(next_states)

            # Bellman backup
            q1_targets = self.q1_target.forward(next_states, next_actions)
            q2_targets = self.q2_target.forward(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * (torch.min(q1_targets, q2_targets) - self.alpha * pi_log_probs)

        # compute actor losses
        q1 = self.q1.forward(states, actions)
        q2 = self.q2.forward(states, actions)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        return q_loss

    def update_parameters(self, memory, batch_size):
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # update actor
        q_params = concatenate(self.q1.parameters(), self.q2.parameters())
        q_optimizer = torch.optim.Adam(q_params, lr=self.lr)
        q_optimizer.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, dones, next_states)
        q_loss.backward()
        q_optimizer.step()

        # update critic
        pi_params = self.policy.parameters()
        pi_loss = self.get_policy_loss(states)
        pi_optimizer = torch.optim.Adam(pi_params, lr=self.lr)
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        # update target networks through Polyak averaging


