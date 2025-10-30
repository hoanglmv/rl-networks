import numpy as np
from environment import WirelessNetworkEnv
from agent import DQNAgent

# Hyperparameters
EPISODES = 1000
STEPS_PER_EPISODE = 100
TARGET_UPDATE_FREQ = 10

if __name__ == "__main__":
    env = WirelessNetworkEnv()
    state_size = env.observation_space_n
    action_size = env.action_space_n
    agent = DQNAgent(state_size, action_size)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay() # Train the agent after each step

            if done:
                break
        
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        print(f"Episode: {e}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    print("Training finished.")