# replay_buffer.py
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr]      = state
        self.action_buf[self.ptr]     = action
        self.reward_buf[self.ptr]     = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr]       = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.state_buf[idxs],
                self.action_buf[idxs],
                self.reward_buf[idxs].flatten(),
                self.next_state_buf[idxs],
                self.done_buf[idxs].flatten())
