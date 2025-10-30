# main.py
import numpy as np
from environment import WirelessEnv
from agent import TD3Agent
from utils import set_seed

def main():
    set_seed(0)
    env = WirelessEnv()
    state_dim = env._get_state().shape[0]
    action_dim = env.K * 2  # 2 thresholds per cell
    agent = TD3Agent(state_dim, action_dim)

    n_episodes = 100
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            agent.train()
            state = next_state
            episode_reward += reward
        print(f"Episode {ep}: Reward = {episode_reward:.2f}")

if __name__ == "__main__":
    main()
