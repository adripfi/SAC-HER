import torch
from itertools import chain as concatenate
import numpy as np
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
import copy
from model import Actor, Critic


class Agent:
    def __init__(self, state_dim, action_dim, action_high, device, lr, gamma=0.99, tau=0.005, alpha=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # two critics + target networks to mitigate overestimation bias using (clipped) double q trick
        self.q1 = Critic(num_inputs=action_dim+state_dim)
        self.q2 = Critic(num_inputs=action_dim+state_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_optimizer = torch.optim.Adam(concatenate(self.q1.parameters(), self.q2.parameters()), lr=self.lr)

        self.policy = Actor(input_size=self.state_dim, output_size=self.action_dim, max_action=self.action_high)
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

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
            # sample target actions from current policy $\alpha$
            next_actions, pi_log_probs = self.policy(next_states)

            # Bellman backup
            q1_targets = self.q1_target.forward(next_states, next_actions)
            q2_targets = self.q2_target.forward(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * (torch.min(q1_targets, q2_targets) - self.alpha * pi_log_probs)

        # compute actor losses
        q1 = self.q1.forward(states, actions)
        q2 = self.q2.forward(states, actions)
        # TODO: check if summing up losses this way is correct
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        return q_loss

    def soft_update(self, source, target):
        # update target weights through Polyak averaging
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def update_parameters(self, memory, batch_size):
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # update actor
        self.q_optimizer.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, dones, next_states)
        q_loss.backward()
        self.q_optimizer.step()

        # update critic
        self.pi_optimizer.zero_grad()
        pi_loss = self.get_policy_loss(states)
        pi_loss.backward()
        self.pi_optimizer.step()

        # update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)


