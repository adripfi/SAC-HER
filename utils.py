import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import distributions as pyd
import math



class LogUtil:
    def __init__(self, comment=""):
        self.writer = SummaryWriter(comment=comment)

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
        elif type == "success":
            self.writer.add_scalar("reward/success", reward, episode)


    def save_checkpoints(self, agent, memory):
        torch.save(agent.q1.state_dict(), self.q1_path)
        torch.save(agent.q2.state_dict(), self.q2_path)
        torch.save(agent.q1_target.state_dict(), self.q1_target_path)
        torch.save(agent.q2_target.state_dict(), self.q2_target_path)
        torch.save(agent.policy.state_dict(), self.actor_path)
        torch.save(agent.alpha, self.temp_path)

    def load_checkpoints(self, agent, path):
        agent.q1.load_state_dict(torch.load(os.path.join(path, "critic_q1")))
        agent.q2.load_state_dict(torch.load(os.path.join(path, "critic_q2")))
        agent.q1_target.load_state_dict(torch.load(os.path.join(path, "critic_q1_target")))
        agent.q2_target.load_state_dict(torch.load(os.path.join(path, "critic_q2_target")))
        agent.policy.load_state_dict(torch.load(os.path.join(path, "actor")))


def test_agent(agent, env, render=False):
    state_dict = env.reset()
    state = state_dict["observation"]
    goal_desired = state_dict["desired_goal"]
    total_reward = 0
    steps = 0
    done = False
    while not done:
        if render: env.render()
        action = agent.sample(state, goal_desired)
        # next_state, reward, done, _ = env.step(action)
        state_dict, reward, done, info = env.step(action)
        next_state = state_dict["observation"]
        # goal_achieved = state_dict["achieved_goal"]

        total_reward += reward
        steps += 1
        state = next_state

    env.close()

    return total_reward, steps

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu









def load_checkpoint(path):
    raise NotImplementedError