import os
import argparse
import gym
import torch
import numpy as np
from agent import Agent
from replay_buffer import ReplayBuffer
from utils import LogUtil, eval, set_seed
from her import HindsightReplayBuffer
# os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/home/adrian/.mujoco/mujoco200/bin"
import random


def train(agent, env, mem, log, max_steps, start_random, start_learning, updates_per_step, eval_interval, batch_size):
    total_steps = 0
    episodes = 0
    successes = np.zeros(eval_interval)
    while total_steps < max_steps:
        episode_reward = 0
        episode_steps = 0
        done = False
        state_dict = env.reset()
        state = state_dict["observation"]
        goal_desired = state_dict["desired_goal"]

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
                    q1_loss, q2_loss, pi_loss, alpha_loss = agent.update_parameters(mem, batch_size)
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

            mem.add(state, goal_desired, goal_achieved, action, reward, next_state, done, done_no_max)

            state = next_state
            goal_desired = state_dict["desired_goal"]

        episodes += 1
        log.reward(total_steps, episode_reward, "train")
        successes[episodes % eval_interval] = float(info["is_success"])

        if episodes % eval_interval == 0:
            # evaluate current policy
            eval_reward, eval_steps, success = eval(agent, env)
            # mean success rate over the last interval
            successes_mean = np.mean(successes)
            print(f"Steps: {total_steps}, Episodes trained: {episodes}, Eval Reward: {eval_reward}, "
                  f"Mean succ.: {successes_mean}, Episode Steps: {eval_steps}")

            log.reward(total_steps, eval_reward, "eval")
            log.reward(total_steps, successes_mean, "success")
            log.save_checkpoints(agent, mem)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC + HER training')

    parser.add_argument("--env", default="FetchPickAndPlace-v1", type=str, help="env id")
    parser.add_argument("--hidden", default=512, type=int, help="size of hidden layers for all networks")
    parser.add_argument("--alpha", default=1., type=float, help="initial alpha value")
    parser.add_argument("--auto_entropy", default=True, type=bool, help="flag for auto entropy tuning")
    parser.add_argument("--buffer_size", default=1_000_000, type=int, help="size of replay buffer in steps")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size for updating networks in steps")
    parser.add_argument("--tau", default=0.0005, type=float, help="parameter for scaling target network soft update")
    parser.add_argument("--gamma", default=0.95, type=float, help="discount factor")
    parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
    parser.add_argument("--num_updates", default=1, type=int, help="number of updates for each env step")
    parser.add_argument("--max_steps", default=2_000_000, type=int, help="max number of env steps")
    parser.add_argument("--random_steps", default=10_000, type=int, help="number of random steps until policy is used")
    parser.add_argument("--start_learning", default=10_000, type=int, help="number of steps until learning begins")
    parser.add_argument("--eval_interval", default=100, type=int, help="interval for policy eval in episodes")
    parser.add_argument("--seed", default=8, type=int, help="seed")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        type=str, help="'cpu' or 'cuda'")

    args = parser.parse_args()
    print(f"Using device {args.device} \n")

    env = gym.make(args.env)
    set_seed(args.seed, args.device, env)

    action_high = env.action_space.high[0]
    state_size = env.observation_space['observation'].shape[0]
    goal_des_size = env.observation_space['desired_goal'].shape[0]
    action_size = env.action_space.shape[0]

    agent = Agent(state_size, action_size, goal_des_size, action_high,
                  args.device, args.lr, args.hidden, args.gamma, args.tau, args.alpha, args.auto_entropy)
    memory = HindsightReplayBuffer(False, env, 1, env.observation_space, env.action_space, args.buffer_size,
                                   args.device)
    log = LogUtil(args.env)

    train(agent, env, memory, log, args.max_steps, args.random_steps, args.start_learning, args.num_updates,
          args.eval_interval, args.batch_size)
