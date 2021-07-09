import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, size, state_dim, action_dim, device):
        self.max_size = size
        self.curr_size = 0
        self.ptr = 0
        self.device = device

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # if buffer is full replace oldest entries
        self.ptr = (self.ptr + 1) % self.max_size
        # cap curr_size at max_size
        self.curr_size = np.min((self.curr_size + 1, self.max_size))

    def sample(self, batch_size):
        # sample batch_size random idx
        indices = np.random.randint(low=0, high=self.curr_size, size=batch_size)

        states = torch.as_tensor(self.states[indices], dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32).to(self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.curr_size



