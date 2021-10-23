import os

import gym
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
seed = 8

env_id = "FetchReach-v1"

env = gym.make(env_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device} \n")
# device = "cpu"

if seed is not None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)



def train():
    action_high = env.action_space.high[0]
    state_size = env.observation_space['observation'].shape[0]
    goal_des_size = env.observation_space['desired_goal'].shape[0]
    action_size = env.action_space.shape[0]

    agent = Agent(state_size, action_size, goal_des_size, action_high, device, lr, hidden_size, gamma, tau, alpha, auto_entropy)
    # TODO: clean memory init i.e. remove redundant action, state and des goal dim. computation
    memory = HindsightReplayBuffer(False, env, 1, env.observation_space, env.action_space, buffer_size, device)
    log = LogUtil(env_id)

    total_steps = 0
    episodes = 0
    while total_steps < max_steps:
        episode_reward = 0
        episode_steps = 0
        done = False
        state_dict = env.reset()
        state = state_dict["observation"]
        goal_desired = state_dict["desired_goal"]
        # successes = np.zeros(eval_interval)

        while not done:
            if total_steps < start_random:
                # sample random action
                action = env.action_space.sample()
            else:
                # sample action from current policy
                action = agent.sample(state, goal_desired)

            if total_steps > start_learning:
                # update agent's weights
                for i in range(updates_per_step):
                    q1_loss, q2_loss, pi_loss, alpha_loss = agent.update_parameters(memory, batch_size)
                    log.loss(q1_loss, q2_loss, pi_loss, alpha_loss, agent.alpha, total_steps)

            state_dict, reward, done, info = env.step(action)
            next_state = state_dict["observation"]
            goal_achieved = state_dict["achieved_goal"]

            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # allow infinite bootstrapping when the episode terminated due to step limit
            done = float(done)
            done_no_max = 0 if episode_steps + 1 == env.spec.max_episode_steps else done

            memory.add(state, goal_desired, goal_achieved, action,  reward, next_state, done, done_no_max)

            state = next_state
            goal_desired = state_dict["desired_goal"]

        episodes += 1
        log.reward(total_steps, episode_reward, "train")
        # successes[episodes % eval_interval] = float(info["is_success"])

        if episodes % eval_interval == 0:
            # evaluate current policy
            eval_reward, eval_steps, success = eval(agent, env)

            # successes_mean = np.mean(successes)
            print(f"Steps: {total_steps}, Episodes trained: {episodes}, Eval Reward: {eval_reward}, Succ: {success}, Episode Steps: {eval_steps}")
            log.reward(total_steps, eval_reward, "eval")
            log.reward(total_steps, success, "success")
            log.save_checkpoints(agent, memory)

    env.close()


if __name__ == "__main__":
    train()
