import numpy as np
import random

class WirelessNetworkEnv:
    def __init__(self, num_cells=3, max_users_per_cell=10, cell_capacity=5, 
                 power_on_cost=1.0, qos_penalty=5.0, user_arrival_rate=0.2, 
                 user_departure_rate=0.1):
        
        self.num_cells = num_cells
        self.max_users_per_cell = max_users_per_cell
        self.cell_capacity = cell_capacity
        self.power_on_cost = power_on_cost
        self.qos_penalty = qos_penalty
        self.user_arrival_rate = user_arrival_rate
        self.user_departure_rate = user_departure_rate

        self.users_in_cells = np.zeros(num_cells, dtype=int)  # Number of active users in each cell
        self.cell_status = np.ones(num_cells, dtype=int)     # 1 for on, 0 for off

        self.action_space_n = 2**num_cells # Each cell can be on or off
        self.observation_space_n = num_cells * 2 # users_in_cells + cell_status

    def reset(self):
        self.users_in_cells = np.array([random.randint(0, self.max_users_per_cell) for _ in range(self.num_cells)])
        self.cell_status = np.ones(self.num_cells, dtype=int) # All cells on initially
        return self._get_state()

    def step(self, action_idx):
        # Convert action index to binary array (e.g., 0 -> [0,0], 1 -> [0,1], 2 -> [1,0], 3 -> [1,1])
        action_binary = np.array([(action_idx >> i) & 1 for i in range(self.num_cells)])
        self.cell_status = action_binary

        self._simulate_user_dynamics()

        reward = self._calculate_reward()
        next_state = self._get_state()
        done = False # For now, the episode never ends

        return next_state, reward, done, {}

    def _get_state(self):
        return np.concatenate((self.users_in_cells, self.cell_status))

    def _calculate_reward(self):
        energy_cost = np.sum(self.cell_status * self.power_on_cost)
        
        qos_violation_penalty = 0
        for i in range(self.num_cells):
            if self.cell_status[i] == 1: # If cell is on
                # Users exceeding capacity in an active cell suffer QoS degradation
                if self.users_in_cells[i] > self.cell_capacity:
                    qos_violation_penalty += (self.users_in_cells[i] - self.cell_capacity) * self.qos_penalty
            else: # If cell is off, all users in it suffer QoS degradation
                qos_violation_penalty += self.users_in_cells[i] * self.qos_penalty
                
        # Reward is negative of total cost (energy + qos penalty)
        reward = - (energy_cost + qos_violation_penalty)
        return reward

    def _simulate_user_dynamics(self):
        for i in range(self.num_cells):
            # User arrivals
            if random.random() < self.user_arrival_rate:
                self.users_in_cells[i] = min(self.users_in_cells[i] + 1, self.max_users_per_cell)
            
            # User departures
            if random.random() < self.user_departure_rate:
                self.users_in_cells[i] = max(self.users_in_cells[i] - 1, 0)

        # Simple user handover (optional, can be made more complex)
        # For now, let's assume users stay in their cells or depart/arrive.
        # More complex handover logic would involve checking neighboring cells and their load.
