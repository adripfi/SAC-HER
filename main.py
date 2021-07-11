import gym
import torch
import numpy as np
from agent import Agent
from replay_buffer import ReplayBuffer
from utils import LogUtil, test_agent


hidden_size = 256
alpha = 0.2
auto_entropy = True
buffer_size = int(1e6)
batch_size = 256
tau = 0.005
gamma = 0.99
lr = 3e-4
updates_per_step = 1
max_steps = int(1e6)
max_episodes = int(1e6)
start_random = 2000
eval_interval = 20  # episodes
seed = None

env_id = "Humanoid-v2"
env = gym.make(env_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device} \n")
# device = "cpu"

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    agent = Agent(state_size, action_size, action_high, device, lr, hidden_size, gamma, tau, alpha, auto_entropy)
    memory = ReplayBuffer(buffer_size, state_size, action_size, device)
    log = LogUtil(env_id)

    updates = 0
    for episodes in range(max_episodes):
        episode_reward = 0
        episode_steps = 0
        total_steps = 0
        done = False
        state = env.reset()

        while not done:
            if total_steps < start_random:
                # sample random action
                action = env.action_space.sample()
            else:
                # sample action from current policy
                action = agent.sample(state)

            if len(memory) > batch_size:
                # update agent's weights
                for i in range(updates_per_step):
                    q1_loss, q2_loss, pi_loss, alpha_loss = agent.update_parameters(memory, batch_size)
                    log.loss(q1_loss, q2_loss, pi_loss, alpha_loss, agent.alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)

            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # allow infinite bootstrapping when the episode terminated due to time limit
            done = False if episode_steps == env._max_episode_steps else done

            memory.push(state, action,  reward, next_state, done)

            state = next_state

        log.reward(total_steps, episode_reward, "train")

        if not episodes % eval_interval:
            # evaluate current policy
            eval_reward, eval_steps = test_agent(agent, env)

            print(f"Episodes trained: {episodes}, Eval Reward: {eval_reward}, Episode Steps: {eval_steps}")
            log.reward(episodes, eval_reward, "eval")
            log.save_checkpoints(agent, memory)

    env.close()


if __name__ == "__main__":
    main()
