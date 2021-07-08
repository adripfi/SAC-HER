import gym
import torch
from agent import Agent
from replay_buffer import ReplayBuffer
import numpy as np

max_episode = 400
hidden_size = 256
alpha = 0.2
buffer_size = int(10e6)
batch_size = 128
tau = 0.002
gamma = 0.99
lr = 1e-3
updates_per_step = 1
max_steps = 1e6
start_random = 2000
env = gym.make("LunarLanderContinuous-v2")


def main():

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # device = "cpu"

    agent = Agent(state_size, action_size, action_high, device, lr, hidden_size, gamma, tau, alpha)
    print(env._max_episode_steps )
    memory = ReplayBuffer(buffer_size, state_size, action_size, device)

    total_steps = 0
    updates = 0

    for episode in range(max_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if total_steps < start_random:
                # sample random action
                action = env.action_space.sample()
            else:
                # sample action from current policy
                action = agent.sample(state)

            # action = agent.sample(state)

            if len(memory) > batch_size:
                for i in range(updates_per_step):
                    agent.update_parameters(memory, batch_size)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            done = False if episode_steps == env._max_episode_steps else done

            memory.push(state, action,  reward, next_state, done)  # Append transition to memory

            state = next_state

        print(f"Episode: {episode}, Reward: {episode_reward}, Total Steps: {total_steps}")
        if total_steps > max_steps:
            break
    env.close()
    test(agent)
    return  agent

def test(agent):
    state  = env.reset()
    total_reward = 0
    for _ in range (1000):
        env.render()
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += 0
        state = next_state
        if done:
            break
    env.close()





if __name__ == "__main__":
    agent = main()
