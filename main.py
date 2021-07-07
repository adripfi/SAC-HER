import gym
import torch
from agent import Agent
from replay_buffer import ReplayBuffer

max_episode = 500
hidden_size = 256
alpha = 0.2
buffer_size = int(10e6)
batch_size = 32
tau = 0.005
gamma = 0.99
lr = 3e-4
updates_per_step = 1
max_steps = 1e6
start_random = 1000

def main():
    env = gym.make("Pendulum-v0")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    agent = Agent(state_size, action_size, action_high, device, lr, hidden_size, gamma, tau, alpha)
    print(buffer_size, state_size)
    memory = ReplayBuffer(buffer_size, state_size, action_size, device)

    total_steps = 0
    updates = 0

    for episode in range(max_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            # sample action from policy
            # TODO: add random steps
            action = agent.sample(state)

            if len(memory) > batch_size:
                for i in range(updates_per_step):
                    agent.update_parameters(memory, batch_size)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        print(f"Episode: {episode}, Reward: {episode_reward}")
        if total_steps > max_steps:
            break
    env.close()



if __name__ == "__main__":
    main()
