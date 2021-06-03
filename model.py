import torch
from torch import nn
from torch.distributions import Normal


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.q_network = nn.Sequential(
            nn.Linear(self.num_inputs + num_actions, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        return self.q_network(x)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_action=1, reparam_noise=1e-6):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # environment specific scaling factor for actions
        self.max_action = torch.tensor(max_action)
        # noise to prevent taking the log of 0
        self.reparam_noise = reparam_noise

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, self.output_size)
        self.sigma = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = nn.ReLU(x)
        x = self.fc2(x)
        x = nn.ReLU(x)

        mu = self.mu(x)
        sigma = self.sigma(x)
        # clip sigma values to reduce width of distribution
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample(self, state):
        mu, sigma = self.forward(state)
        # sample from normal distribution and add noise for reparametrization trick
        prob = Normal(mu, sigma)
        actions = prob.rsample()

        # squash gaussian and scale action beyond +/- 1
        actions = torch.tanh(actions) * self.max_action

        # log probability of of actions for loss function
        log_probs = prob.log_prob(actions)
        # enforce action bounds as proposed by the authors in appendix
        log_probs -= torch.log(1 - actions.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdims=True)

        return actions, log_probs
