
# simulation/main.py
import numpy as np
import torch

from config import *
from environment import NetworkEnvironment
from agent import TD3Agent
from logger import SimulationLogger

def main():
    """Main function to run the DRL simulation."""
    env = NetworkEnvironment()
    logger = SimulationLogger()

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = float(ACTION_MAX)

    agent = TD3Agent(state_dim, action_dim, max_action)

    # --- Main Training Loop ---
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        episode_power = 0
        episode_throughput = 0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # Select action with exploration noise
            action = agent.select_action(state)
            action = (action + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)

            # Perform action
            # We control the first sector of the first eNB for simplicity
            next_state, reward, done = env.step(enb_id=0, sector_id=0, action=action)

            # Store data in replay buffer
            agent.replay_buffer.add(state, action, next_state, reward, float(done))

            # Train agent
            if agent.replay_buffer.size > REPLAY_BUFFER_SIZE:
                agent.train()

            state = next_state
            episode_reward += reward
            
            # Collect metrics
            total_power = sum(enb.get_total_power_consumption() for enb in env.enbs)
            total_throughput = sum(ue.throughput_mbps for ue in env.ues)
            episode_power += total_power
            episode_throughput += total_throughput


            if done:
                break
        
        avg_power = episode_power / step
        avg_throughput = episode_throughput / step
        logger.log_metrics(episode, step, episode_reward, avg_power, avg_throughput)

        # Save model periodically
        if episode % 100 == 0:
            agent.save(f"models/td3_model_episode_{episode}")
            logger.log(f"Saved model at episode {episode}")

if __name__ == "__main__":
    # Create a directory for saving models if it doesn't exist
    import os
    os.makedirs("models", exist_ok=True)
    main()
