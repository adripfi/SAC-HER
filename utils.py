import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

class LogUtil:
    def __init__(self):
        self.writer = SummaryWriter()

        self.path_checkpoints = os.path.join("models", os.path.split(self.writer.file_writer.event_writer._logdir)[-1])
        if not os.path.exists(self.path_checkpoints):
            os.makedirs(self.path_checkpoints)

        self.q1_path = os.path.join(self.path_checkpoints, "critic_q1")
        self.q2_path = os.path.join(self.path_checkpoints, "critic_q2")
        self.q1_target_path = os.path.join(self.path_checkpoints, "critic_q1_target")
        self.q2_target_path = os.path.join(self.path_checkpoints, "critic_q2_target")
        self.actor_path = os.path.join(self.path_checkpoints, "actor")
        self.temp_path = os.path.join(self.path_checkpoints, "alpha")
        self.memory_path = os.path.join(self.path_checkpoints, "memory.p")


    def loss(self, q1_loss, q2_loss, pi_loss, alpha_loss, alpha, step):
        self.writer.add_scalar("loss/q1", q1_loss, step)
        self.writer.add_scalar("loss/q2", q2_loss, step)
        self.writer.add_scalar("loss/pi", pi_loss, step)
        self.writer.add_scalar("loss/alpha", alpha_loss, step)
        self.writer.add_scalar("param/alpha", alpha, step)

    def reward(self, episode, reward, type):
        if type == "train":
            self.writer.add_scalar("reward/train", reward, episode)
        elif type == "eval":
            self.writer.add_scalar("reward/eval", reward, episode)

    def save_checkpoints(self, agent, memory):
        torch.save(agent.q1.state_dict(), self.q1_path)
        torch.save(agent.q2.state_dict(), self.q2_path)
        torch.save(agent.q1_target.state_dict(), self.q1_target_path)
        torch.save(agent.q2_target.state_dict(), self.q2_target_path)
        torch.save(agent.policy.state_dict(), self.actor_path)
        torch.save(agent.alpha, self.temp_path)

        with open(self.memory_path, "wb") as mem:
            pickle.dump(memory, mem)

    def load_checkpoints(self, agent, path):
        agent.q1.load_state_dict(torch.load(os.path.join(path, "critic_q1")))
        agent.q2.load_state_dict(torch.load(os.path.join(path, "critic_q2")))
        agent.q1_target.load_state_dict(torch.load(os.path.join(path, "critic_q1_target")))
        agent.q2_target.load_state_dict(torch.load(os.path.join(path, "critic_q2_target")))
        agent.policy.load_state_dict(torch.load(os.path.join(path, "actor")))

        with open(os.path.join(path, "memory.p"), "rb") as mem:
            memory = pickle.load(mem)




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









def load_checkpoint(path):
    raise NotImplementedError