import numpy as np

class User:
    def __init__(self, user_id, position, traffic_demand):
        self.user_id = user_id
        self.position = position  # (x, y) coordinates
        self.traffic_demand = traffic_demand # Data rate demand
        self.scheduled_cell = None
        self.buffered_data_size = traffic_demand # Simplified for now

class Cell:
    def __init__(self, cell_id, cell_type, frequency_band, bandwidth, prbs_total, enb_id, sector_id,
                 tx_power, noise_power_density):
        self.cell_id = cell_id
        self.cell_type = cell_type # K types
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
        self.prbs_total = prbs_total # Mk in the paper
        self.enb_id = enb_id
        self.sector_id = sector_id
        self.users = [] # Users currently scheduled in this cell
        self.mode = 'active' # 'active' or 'standby'
        self.prbs_used = 0 # Nk,i,j in the paper
        self.offset_power = 0 # Pk^0 in the paper
        self.variable_power = 0 # Pk^v in the paper
        self.low_power_mode_power = 0 # Pk^e in the paper
        self.tx_power = tx_power # P_Tx in the paper
        self.noise_power_density = noise_power_density # No in the paper

    def calculate_load(self):
        if self.prbs_total > 0:
            return self.prbs_used / self.prbs_total
        return 0

    def calculate_power_consumption(self):
        if self.mode == 'active':
            mk = 1
        else:
            mk = 0 # standby mode
        
        load = self.calculate_load()
        # Equation (3) from the paper: P_i,j = (P_k^0 + P_k^v * load) * m_k + P_k^e * (1 - m_k)
        return (self.offset_power + self.variable_power * load) * mk + self.low_power_mode_power * (1 - mk)

    def get_channel_gain(self, user_position):
        # Simplified channel gain (path loss only, no fading for now)
        # Distance-based path loss model
        cell_position = self.get_position() # Assuming cell has a position
        distance = np.sqrt((cell_position[0] - user_position[0])**2 + (cell_position[1] - user_position[1])**2)
        if distance < 1: # Avoid division by zero or very small distances
            distance = 1
        
        # A simple path loss model: L = 128.1 + 37.6 * log10(distance_km)
        # Convert distance to km for this model, assuming position is in meters
        distance_km = distance / 1000
        path_loss_db = 128.1 + 37.6 * np.log10(distance_km)
        path_loss_linear = 10**(-path_loss_db / 10)
        return path_loss_linear

    def get_position(self):
        # Placeholder: In a real scenario, cell positions would be defined
        # For now, let's assume cells are at the eNB position for simplicity
        # and add a small offset for different sectors/cells if needed later
        return (0,0) # This needs to be refined when eNB positions are used

    def calculate_sinr(self, user, interfering_cells):
        # Equation (4) from the paper: SINR = (P_Tx * g_k,i,j,u) / (N_0 * B_W + sum(P_Tx * g_k,p,q,u))
        
        # Desired signal power
        channel_gain_desired = self.get_channel_gain(user.position)
        desired_power = self.tx_power * channel_gain_desired

        # Noise power (N_0 * B_W)
        noise_power = self.noise_power_density * self.bandwidth

        # Interference power
        interference_power = 0
        for interfering_cell in interfering_cells:
            if interfering_cell.mode == 'active': # Only active cells interfere
                channel_gain_interfering = interfering_cell.get_channel_gain(user.position)
                interference_power += interfering_cell.tx_power * channel_gain_interfering
        
        sinr = desired_power / (noise_power + interference_power)
        return sinr

    def calculate_throughput(self, user, sinr, assigned_prbs):
        # Equation (5) and (6) from the paper (simplified for now)
        # d_k,i,j,u = min(n_k,i,j,u * log2(1 + SINR_k,i,j,u), V_k,i,j,u)
        # Assuming assigned_prbs is n_k,i,j,u and user.buffered_data_size is V_k,i,j,u
        
        # Shannon capacity formula for spectral efficiency
        spectral_efficiency = np.log2(1 + sinr) # bits/s/Hz
        
        # Throughput = assigned_prbs * (bandwidth / prbs_total) * spectral_efficiency
        # This is a simplification. The paper uses n_k,i,j,u * log2(1 + SINR_k,i,j,u)
        # which implies that n_k,i,j,u is a factor of bandwidth.
        # Let's assume each PRB contributes equally to bandwidth
        prb_bandwidth = self.bandwidth / self.prbs_total
        
        data_rate_per_prb = prb_bandwidth * spectral_efficiency
        
        # The paper's formula is a bit ambiguous with units, assuming n_k,i,j,u is number of PRBs
        # and the log2 term gives bits/symbol or similar.
        # Let's use a common interpretation: throughput is sum of data rates per PRB
        # For simplicity, assume assigned_prbs directly scales the throughput
        throughput_from_prbs = assigned_prbs * data_rate_per_prb
        
        # The min with buffered_data_size implies that a user can only receive up to its demand
        return min(throughput_from_prbs, user.buffered_data_size)


class Sector:
    def __init__(self, sector_id, enb_id, num_cell_types, cell_params):
        self.sector_id = sector_id
        self.enb_id = enb_id
        self.cells = []
        for k in range(num_cell_types):
            cell_id = f"eNB{enb_id}_Sec{sector_id}_Cell{k}"
            params = cell_params[k]
            cell = Cell(cell_id, k, params['frequency_band'], params['bandwidth'], 
                        params['prbs_total'], enb_id, sector_id, params['tx_power'], 
                        params['noise_power_density'])
            self.cells.append(cell)

class eNB:
    def __init__(self, enb_id, position, num_sectors=3, num_cell_types=4, cell_params=None):
        self.enb_id = enb_id
        self.position = position # (x, y) coordinates
        self.sectors = []
        if cell_params is None:
            # Default cell parameters if not provided
            cell_params = [
                {'frequency_band': 2000, 'bandwidth': 10e6, 'prbs_total': 100, 'tx_power': 0.1, 'noise_power_density': 1e-18},
                {'frequency_band': 2100, 'bandwidth': 15e6, 'prbs_total': 150, 'tx_power': 0.1, 'noise_power_density': 1e-18},
                {'frequency_band': 2200, 'bandwidth': 20e6, 'prbs_total': 200, 'tx_power': 0.1, 'noise_power_density': 1e-18},
                {'frequency_band': 2300, 'bandwidth': 25e6, 'prbs_total': 250, 'tx_power': 0.1, 'noise_power_density': 1e-18},
            ]

        for i in range(num_sectors):
            sector = Sector(i, enb_id, num_cell_types, cell_params)
            self.sectors.append(sector)
        self.users = [] # Users associated with this eNB

    def add_user(self, user):
        self.users.append(user)

    def remove_user(self, user):
        if user in self.users:
            self.users.remove(user)

    def get_all_cells(self):
        all_cells = []
        for sector in self.sectors:
            all_cells.extend(sector.cells)
        return all_cells

# Example Usage (for testing the classes)
if __name__ == "__main__":
    # Define cell parameters for different cell types
    cell_parameters = [
        {'frequency_band': 2000e6, 'bandwidth': 10e6, 'prbs_total': 100, 'tx_power': 0.1, 'noise_power_density': 1e-18}, # Cell Type 0
        {'frequency_band': 2100e6, 'bandwidth': 15e6, 'prbs_total': 150, 'tx_power': 0.1, 'noise_power_density': 1e-18}, # Cell Type 1
        {'frequency_band': 2200e6, 'bandwidth': 20e6, 'prbs_total': 200, 'tx_power': 0.1, 'noise_power_density': 1e-18}, # Cell Type 2
        {'frequency_band': 2300e6, 'bandwidth': 25e6, 'prbs_total': 250, 'tx_power': 0.1, 'noise_power_density': 1e-18}, # Cell Type 3
    ]

    # Create an eNB
    enb1 = eNB(enb_id=1, position=(0, 0), num_cell_types=len(cell_parameters), cell_params=cell_parameters)
    print(f"eNB {enb1.enb_id} created at {enb1.position}")

    # Create another eNB for interference
    enb2 = eNB(enb_id=2, position=(1000, 0), num_cell_types=len(cell_parameters), cell_params=cell_parameters)

    # Access sectors and cells
    all_network_cells = []
    for sector in enb1.sectors:
        for cell in sector.cells:
            all_network_cells.append(cell)
    for sector in enb2.sectors:
        for cell in sector.cells:
            all_network_cells.append(cell)

    # Create a user
    user1 = User(user_id=101, position=(50, 50), traffic_demand=5e6) # 5 Mbps demand

    # Simulate some PRB usage and power consumption
    for sector in enb1.sectors:
        for cell in sector.cells:
            print(f"    Cell {cell.cell_id}, Type: {cell.cell_type}, PRBs: {cell.prbs_total}, Mode: {cell.mode}")
            cell.prbs_used = 50 # Example PRB usage
            print(f"      Load: {cell.calculate_load():.2f}")
            
            cell.offset_power = 10 # Placeholder
            cell.variable_power = 0.5 # Placeholder
            cell.low_power_mode_power = 2 # Placeholder
            print(f"      Power Consumption: {cell.calculate_power_consumption():.2f} W")
            
            # Test SINR and Throughput
            # For interfering cells, exclude the current cell itself
            interfering_cells = [c for c in all_network_cells if c.cell_id != cell.cell_id]
            sinr = cell.calculate_sinr(user1, interfering_cells)
            print(f"      SINR for user1: {sinr:.2e}")
            
            assigned_prbs = 20 # Example assigned PRBs for user1
            throughput = cell.calculate_throughput(user1, sinr, assigned_prbs)
            print(f"      Throughput for user1: {throughput:.2e} bps")

            cell.mode = 'standby'
            print(f"      Power Consumption (standby): {cell.calculate_power_consumption():.2f} W")

    print(f"User {user1.user_id} added to eNB {enb1.enb_id}")
    
    all_cells_enb1 = enb1.get_all_cells()
    print(f"Total cells in eNB1: {len(all_cells_enb1)}")
