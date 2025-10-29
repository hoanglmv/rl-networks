
# simulation/environment.py
import numpy as np
from collections import deque

from config import *
from network_objects import eNodeB, UserEquipment

class NetworkEnvironment:
    """Simulates the wireless network environment."""
    def __init__(self):
        self.enb_positions = self._generate_enb_positions()
        self.enbs = [eNodeB(i, self.enb_positions[i]) for i in range(NUM_ENB)]
        self.ues = self._initialize_ues()
        self.time = 0  # Simulation time in seconds

        # State definition (as per paper for one eNB, one sector)
        # For simplicity, we control one sector of the first eNB.
        # State: [hour, day, cell_modes, sector_one_hot, cell_loads]
        self.num_cells = NUM_CELL_TYPES
        # Past loads for 4 previous decision periods + current average load
        self.load_history_len = 5
        self.state_dim = 1 + 1 + self.num_cells + SECTORS_PER_ENB + (self.num_cells * self.load_history_len)
        self.action_dim = self.num_cells * 2 # 2 thresholds (activation, deactivation) per cell

        # Store load history for state representation
        self.load_history = {enb_id: {sector_id: {cell_type: deque(maxlen=self.load_history_len)
                                                  for cell_type in CARRIER_BANDS}
                                     for sector_id in range(SECTORS_PER_ENB)}
                             for enb_id in range(NUM_ENB)}

    def _generate_enb_positions(self):
        """Generates positions for eNBs in a Hex 7 deployment."""
        positions = []
        # Center eNB at (0,0)
        positions.append(np.array([0.0, 0.0]))

        # Six eNBs around the center in a hexagonal pattern
        for i in range(6):
            angle = 2 * np.pi / 6 * i
            x = HEX_GRID_RADIUS_M * np.cos(angle)
            y = HEX_GRID_RADIUS_M * np.sin(angle)
            positions.append(np.array([x, y]))
        return positions

    def _initialize_ues(self):
        ues = []
        # Initialize UEs within the hexagonal grid boundaries
        # For simplicity, a square bounding box around the hex grid
        min_x = -HEX_GRID_RADIUS_M * 1.5
        max_x = HEX_GRID_RADIUS_M * 1.5
        min_y = -HEX_GRID_RADIUS_M * 1.5
        max_y = HEX_GRID_RADIUS_M * 1.5

        for i in range(NUM_ENB * TOTAL_UES_PER_ENB):
            is_static = i < (NUM_ENB * STATIC_UES_PER_ENB)
            position = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            ues.append(UserEquipment(ue_id=i, is_static=is_static, position=position))
        return ues

    def _calculate_distance(self, pos1, pos2):
        """Calculates Euclidean distance between two 2D points."""
        return np.linalg.norm(pos1 - pos2)

    def _get_state(self, enb_id, sector_id):
        """Constructs the state vector for a given agent (eNB sector)."""
        sector = self.enbs[enb_id].sectors[sector_id]

        # Time features
        hour_of_day = (self.time / 3600) % 24
        day_of_week = (self.time / (3600 * 24)) % 7

        # Cell modes (on/off)
        cell_modes = [cell.is_active for cell in sector.cells.values()]

        # Sector one-hot encoding
        sector_one_hot = np.zeros(SECTORS_PER_ENB)
        sector_one_hot[sector_id] = 1

        # Cell loads (historical)
        cell_loads = []
        for cell_type in CARRIER_BANDS:
            history = list(self.load_history[enb_id][sector_id][cell_type])
            # Pad with zeros if history is not full
            padded_history = history + [0] * (self.load_history_len - len(history))
            cell_loads.extend(padded_history)

        state = np.concatenate(
            [[hour_of_day / 24.0, day_of_week / 7.0]] +
            [cell_modes] +
            [sector_one_hot] +
            [cell_loads]
        ).flatten()

        return state

    def _calculate_reward(self, enb_id):
        """
        Calculates the reward for a given eNB based on the paper's formula (Eq. 11).
        R_t = alpha * sum(log(throughput_u)) - beta * P_total
        With a penalty if any user's throughput is below the minimum requirement.
        """
        enb = self.enbs[enb_id]
        sum_user_utility = 0
        num_associated_ues = 0

        # Check throughput constraint and calculate utility
        for ue in self.ues:
            # Consider only UEs associated with the current eNB
            if ue.serving_cell and ue.serving_cell.sector.enb == enb:
                num_associated_ues += 1
                # Penalty if throughput is below the minimum requirement
                if ue.throughput_mbps < MIN_THROUGHPUT_MBPS:
                    return -10.0

                # Add user utility (log of throughput)
                # Add a small epsilon to avoid log(0)
                sum_user_utility += np.log(ue.throughput_mbps + 1e-9)

        # If no UEs are associated, the utility is 0
        if num_associated_ues == 0:
            sum_user_utility = 0

        # Get total power consumption for the eNB
        total_power = enb.get_total_power_consumption()

        # Calculate final reward using formula from the paper
        reward = (REWARD_ALPHA * sum_user_utility) - (REWARD_BETA * total_power)

        return reward

    def _update_network_state(self):
        """Simulates network activity for one metric gathering period."""
        # 1. UE movement
        for ue in self.ues:
            ue.update_position()

        # 2. Traffic generation (simplified)
        # In a real sim, this would follow a traffic model (e.g., FTP model from paper)
        for ue in self.ues:
            ue.receive_data(np.random.exponential(0.5)) # Random data arrival

        # 3. UE association and resource allocation (highly simplified)
        # This is a very complex topic. We'll do a naive association.
        for enb in self.enbs:
            for sector in enb.sectors:
                for cell in sector.cells.values():
                    cell.prbs_used = 0
                    cell.users = []

        for ue in self.ues:
            # Naive association: connect to the first available active cell
            # A real implementation uses RSRP maps.
            for enb in self.enbs:
                for sector in enb.sectors:
                    for cell in sector.cells.values():
                        if cell.is_active:
                            ue.serving_cell = cell
                            cell.add_user(ue)
                            break
                    if ue.serving_cell: break
                if ue.serving_cell: break

        # 4. Throughput and PRB calculation (SINR-based)
        # First, gather all active cells in the network
        all_active_cells = []
        for enb in self.enbs:
            for sector in enb.sectors:
                for cell in sector.cells.values():
                    if cell.is_active:
                        all_active_cells.append(cell)

        for enb in self.enbs:
            for sector in enb.sectors:
                for cell in sector.cells.values():
                    cell.prbs_used = 0 # Reset PRBs used for this period
                    if not cell.is_active or not cell.users:
                        for ue in cell.users:
                            ue.throughput_mbps = 0
                        continue

                    # Allocate PRBs proportionally among users in the cell
                    prbs_per_user = cell.prbs_total / len(cell.users)
                    cell.prbs_used = cell.prbs_total # All PRBs are used if cell is active and has users

                    for ue in cell.users:
                        # Calculate distance between UE and serving cell
                        distance_to_serving_cell = self._calculate_distance(ue.position, cell.position)
                        if distance_to_serving_cell == 0: distance_to_serving_cell = 1e-6 # Avoid division by zero

                        # Calculate signal power from serving cell
                        signal_power = P_TX_WATTS[cell.cell_type] / (distance_to_serving_cell ** PATH_LOSS_EXPONENT)

                        # Calculate interference power from other active cells
                        interference_power = 0
                        for other_cell in all_active_cells:
                            if other_cell != cell: # Exclude serving cell itself
                                distance_to_other_cell = self._calculate_distance(ue.position, other_cell.position)
                                if distance_to_other_cell == 0: distance_to_other_cell = 1e-6 # Avoid division by zero
                                interference_power += P_TX_WATTS[other_cell.cell_type] / (distance_to_other_cell ** PATH_LOSS_EXPONENT)

                        # Calculate SINR
                        sinr = signal_power / (NOISE_POWER_WATTS + interference_power)

                        # Calculate spectral efficiency (bits/s/Hz)
                        spectral_efficiency = np.log2(1 + sinr)

                        # Calculate maximum achievable throughput for this UE (Mbps)
                        # Throughput = spectral_efficiency * PRBs * PRB_Bandwidth
                        max_achievable_throughput_mbps = spectral_efficiency * prbs_per_user * PRB_BANDWIDTH_MHZ

                        # Throughput is limited by buffered data (Eq. 5 & 6 from paper)
                        # Convert buffered_data_mb to throughput over METRIC_GATHERING_PERIOD_SEC
                        throughput_from_buffered_data_mbps = (ue.buffered_data_mb * 8) / METRIC_GATHERING_PERIOD_SEC # Convert MB to Mbits

                        ue.throughput_mbps = min(max_achievable_throughput_mbps, throughput_from_buffered_data_mbps)
                        ue.buffered_data_mb = max(0, ue.buffered_data_mb - (ue.throughput_mbps * METRIC_GATHERING_PERIOD_SEC / 8)) # Convert Mbits back to MB

        # 5. Update power consumption for all cells
        for enb in self.enbs:
            for sector in enb.sectors:
                for cell in sector.cells.values():
                    cell.update_power_consumption()

    def step(self, enb_id, sector_id, action):
        """Run one timestep of the environment's dynamics."""
        sector = self.enbs[enb_id].sectors[sector_id]

        # 1. Apply action: Set new thresholds for each cell in the sector
        for i, cell in enumerate(sector.cells.values()):
            cell.deactivation_threshold = action[i * 2]
            cell.activation_threshold = action[i * 2 + 1]

        # 2. Simulate network dynamics over the decision period T
        # The decision period T consists of L metric gathering periods.
        cumulative_reward = 0
        for _ in range(int(METRIC_GATHERING_FACTOR)):
            self.time += METRIC_GATHERING_PERIOD_SEC
            self._update_network_state()

            # Update cell on/off status based on the *average* load over the period
            # For simplicity, we use the current load. A better way is to average.
            for cell_type, cell in sector.cells.items():
                # The paper says the agent decides based on the average load over T.
                # We simplify here and check at each sub-step.
                cell.update_status(cell.load)
                self.load_history[enb_id][sector_id][cell_type].append(cell.load)

            cumulative_reward += self._calculate_reward(enb_id)

        # 3. Get next state
        next_state = self._get_state(enb_id, sector_id)

        # 4. Check if done (e.g., end of simulation time)
        done = self.time >= (SIMULATION_TIME_WEEKS * 7 * 24 * 3600)

        return next_state, cumulative_reward / METRIC_GATHERING_FACTOR, done

    def reset(self):
        """Resets the state of the environment to an initial state."""
        self.time = 0
        self.ues = self._initialize_ues()
        self.enbs = [eNodeB(i) for i in range(NUM_ENB)]
        self.load_history = {enb_id: {sector_id: {cell_type: deque(maxlen=self.load_history_len)
                                                  for cell_type in CARRIER_BANDS}
                                     for sector_id in range(SECTORS_PER_ENB)}
                             for enb_id in range(NUM_ENB)}
        # For simplicity, we return the state of the first sector of the first eNB
        return self._get_state(enb_id=0, sector_id=0)
