
# simulation/utils.py
import numpy as np

def normalize_state(state, state_bounds):
    """Normalizes the state vector to be between -1 and 1."""
    normalized_state = np.zeros_like(state, dtype=np.float32)
    for i, (low, high) in enumerate(state_bounds):
        normalized_state[i] = 2 * ((state[i] - low) / (high - low)) - 1
    return normalized_state

def denormalize_action(normalized_action, action_bounds):
    """Denormalizes the action vector from -1 to 1 to its original range."""
    action = np.zeros_like(normalized_action, dtype=np.float32)
    for i, (low, high) in enumerate(action_bounds):
        action[i] = low + (0.5 * (normalized_action[i] + 1) * (high - low))
    return action

# Add other utility functions as needed, e.g., for plotting, data processing.
