import os
import torch
from torch.utils.tensorboard import SummaryWriter

class LogUtil:
    def __init__(self):
        self.writer = SummaryWriter()

    def loss(self, q1_loss, q2_loss, pi_loss, alpha_loss, step):
        self.writer.add_scalar("loss/q1", q1_loss, step)
        self.writer.add_scalar("loss/q2", q2_loss, step)
        self.writer.add_scalar("loss/pi", pi_loss, step)
        self.writer.add_scalar("loss/alpha", alpha_loss, step)

    def reward(self, episode, reward, type):
        if type == "train":
            self.writer.add_scalar("reward/train", reward, episode)
        elif type == "eval":
            self.writer.add_scalar("reward/eval", reward, episode)


def test_agent(agent, env, render=False):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    while not done:
        if render: env.render()
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

    env.close()

    return total_reward, steps


def save_checkpoint(model, path):
    raise NotImplementedError


def load_checkpoint(path):
    raise NotImplementedError