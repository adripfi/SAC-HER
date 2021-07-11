import torch
from itertools import chain as concatenate
import numpy as np
import torch.nn.functional as F
import copy
from model import Actor, Critic


class Agent:
    def __init__(self, state_dim, action_dim, goal_des_dim, action_high, device, lr, hidden_size=256, gamma=0.99, tau=0.005,
                 alpha=0.2, auto_entropy=True):
        # TODO: change notation to use size instead of dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_des_dim = goal_des_dim
        self.action_high = action_high
        self.device = device
        self.hidden_size = hidden_size

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy

        # two critics + target networks to mitigate overestimation bias using (clipped) double q trick
        self.q1 = Critic(num_inputs=action_dim + state_dim + goal_des_dim, hidden_size=self.hidden_size).to(self.device)
        self.q2 = Critic(num_inputs=action_dim + state_dim + goal_des_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_optimizer = torch.optim.Adam(concatenate(self.q1.parameters(), self.q2.parameters()), lr=self.lr)

        # Gaussian policy
        self.policy = Actor(input_size=self.state_dim + goal_des_dim, output_size=self.action_dim, max_action=self.action_high,
                            hidden_size=self.hidden_size).to(self.device)
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # automatic entropy tuning
        if self.auto_entropy:
            # entropy target heuristic -|A| proposed by the authors
            self.entropy_target = -self.action_dim
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def sample(self, state, goal_des):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        goal_des = torch.as_tensor(goal_des, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy.sample(state, goal_des)

        return action.detach().cpu().numpy()[0]

    def get_policy_loss(self, states, goals_desired):
        next_actions, pi_log_probs = self.policy.sample(states, goals_desired)
        q1_target = self.q1_target.forward(states, goals_desired, next_actions)
        q2_target = self.q2_target.forward(states, goals_desired, next_actions)
        q_target_min = torch.min(q1_target, q2_target)
        pi_loss = (self.alpha * pi_log_probs - q_target_min).mean()

        return pi_loss, pi_log_probs

    def get_q_loss(self, states, goals_desired, actions, rewards, dones, next_states):
        # compute TD target using target q networks
        with torch.no_grad():
            # sample target actions from current policy
            next_actions, pi_log_probs = self.policy.sample(next_states, goals_desired)

            # Bellman backup
            q1_targets = self.q1_target.forward(next_states, goals_desired, next_actions)
            q2_targets = self.q2_target.forward(next_states, goals_desired, next_actions)
            # compute TD target using minimum value of target networks
            # additionally filter tuples at end of episode (done == 1)
            y = rewards + self.gamma * (1 - dones) * (torch.min(q1_targets, q2_targets) - self.alpha * pi_log_probs)

        # compute critic losses
        q1 = self.q1.forward(states, goals_desired, actions)
        q1_loss = F.mse_loss(q1, y)
        q2 = self.q2.forward(states, goals_desired,  actions)
        q2_loss = F.mse_loss(q2, y)

        return q1_loss, q2_loss

    def soft_update(self, source, target):
        # update target weights through Polyak averaging
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def update_parameters(self, memory, batch_size):
        states, goals_desired, actions, rewards, next_states, dones = memory.sample(batch_size)

        # update critics
        self.q_optimizer.zero_grad()
        q1_loss, q2_loss = self.get_q_loss(states, goals_desired, actions, rewards, dones, next_states)
        # TODO: check if summing up losses this way is correct
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q_optimizer.step()

        # update actor
        self.pi_optimizer.zero_grad()
        pi_loss, pi_log = self.get_policy_loss(states, goals_desired)
        pi_loss.backward()
        self.pi_optimizer.step()

        # update critic target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        # update temperature
        if self.auto_entropy:
            # TODO: check sign of log prob
            self.alpha_optimizer.zero_grad()
            alpha_loss = (self.log_alpha * (-pi_log - self.entropy_target).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            # only for logging purposes
            alpha_loss = 0

        return q1_loss, q2_loss, pi_loss, alpha_loss
