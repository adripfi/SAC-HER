import os
import argparse
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







def eval(agent, env):
    state_dict = env.reset()
    state = state_dict["observation"]
    goal_desired = state_dict["desired_goal"]
    total_reward = 0
    steps = 0
    done = False
    while not done:
        action = agent.sample(state, goal_desired)
        state_dict, reward, done, info = env.step(action)
        next_state = state_dict["observation"]

        total_reward += reward
        steps += 1
        state = next_state

    env.close()

    return total_reward, steps, float(info["is_success"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC + HER eva')
    parser.add_argument("--env", default="FetchPickAndPlace-v1", type=str, help="env id")
    env_id = "FetchPickAndPlace-v1"

    env = Monitor(gym.make(env_id), './video', force=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device {device} \n")
    device = "cpu"
    action_high = env.action_space.high[0]
    state_size = env.observation_space['observation'].shape[0]
    goal_des_size = env.observation_space['desired_goal'].shape[0]
    # achieved_goal_shape = env.observation_space[prefix + 'achieved_goal'].shape
    action_size = env.action_space.shape[0]
    # print("action", action_size, "state", state_size, "goal_des", goal_des_size)

    agent = Agent(state_size, action_size, goal_des_size, action_high, device)
    agent.load_checkpoints("/home/adrian/Documents/01_Uni_Stuff/00_SoSe_21/SAC/models/Oct22_15-54-06_WorkstationFetchPickAndPlace-v1")
    eval(agent, env)

