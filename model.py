import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size=256):
        super(Critic, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.num_inputs, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, input_size, output_size, max_action, const_noise=1e-6, max_sigma=2, hidden_size=256): 
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_action = torch.tensor(max_action)
        self.const_noise = const_noise
        self.max_sigma = max_sigma
        self.hidden_size = hidden_size
        # environment specific scaling factor for actions
        # noise to prevent taking the log of 0

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, self.output_size)
        self.sigma = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        mu = self.mu(x)
        sigma = self.sigma(x)
        # clip sigma values to reduce width of distribution
        sigma = torch.clamp(sigma, min=self.const_noise, max=self.max_sigma)

        return mu, sigma

    def sample(self, state):
        mu, sigma = self.forward(state)
        # sample from normal distribution and add noise for reparametrization trick
        prob = Normal(mu, sigma)
        n = prob.rsample()

        # squash gaussian 
        a = torch.tanh(n)

        # get log probs of action sampled
        log_probs = prob.log_prob(n)
        # enforce action bounds as proposed by the authors in the appendix
        log_probs -= torch.log(self.max_action * ( 1 - a.pow(2)) + self.const_noise)
        log_probs = log_probs.sum(1, keepdims=True)

        # scale actions beyond +/-1
        actions = a * self.max_action

        return actions, log_probs
