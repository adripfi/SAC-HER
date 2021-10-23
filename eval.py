import os

import gym
from gym.wrappers import Monitor
import torch
import numpy as np
from agent import Agent
from replay_buffer import ReplayBuffer
from utils import LogUtil, eval
from her import HindsightReplayBuffer
# os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/home/adrian/.mujoco/mujoco200/bin"
import random



hidden_size = 512
alpha = 1.
auto_entropy = True
buffer_size = int(1e6)
batch_size = 1024
tau = 0.0005
gamma = 0.95
lr = 0.0005
updates_per_step = 1
max_steps = int(2e6)
# max_episodes = int(1e6)
start_random = int(1e4)
start_learning = int(1e4)
eval_interval = 100  # episodes
seed = None

env_id = "FetchPickAndPlace-v1"

env = Monitor(gym.make(env_id), './video', force=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device {device} \n")
device = "cpu"

if seed is not None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)


def playback(agent, env, render=False):
    state_dict = env.reset()
    state = state_dict["observation"]
    goal_desired = state_dict["desired_goal"]
    total_reward = 0
    steps = 0
    done = False
    while not done:
        if render:
            env.render()
        action = agent.sample(state, goal_desired)
        state_dict, reward, done, info = env.step(action)
        next_state = state_dict["observation"]

        total_reward += reward
        steps += 1
        state = next_state

    env.close()

    print(f"Steps {steps}, Reward {total_reward}, Succes {info['is_success']}")



if __name__ == "__main__":
    log = LogUtil()
    action_high = env.action_space.high[0]
    state_size = env.observation_space['observation'].shape[0]
    goal_des_size = env.observation_space['desired_goal'].shape[0]
    # achieved_goal_shape = env.observation_space[prefix + 'achieved_goal'].shape
    action_size = env.action_space.shape[0]
    # print("action", action_size, "state", state_size, "goal_des", goal_des_size)

    agent = Agent(state_size, action_size, goal_des_size, action_high, device, lr, hidden_size, gamma, tau, alpha, auto_entropy)
    log.load_checkpoints(agent, "/home/adrian/Documents/01_Uni_Stuff/00_SoSe_21/SAC/models/Oct22_15-54-06_WorkstationFetchPickAndPlace-v1")
    playback(agent, env)

