
# simulation/network_objects.py
import numpy as np
from config import *

class UserEquipment:
    """Represents a single User Equipment (UE)."""
    def __init__(self, ue_id, is_static=True, position=None):
        self.ue_id = ue_id
        self.is_static = is_static
        self.position = position if position is not None else self._initialize_position()
        self.serving_cell = None
        self.buffered_data_mb = 0
        self.throughput_mbps = 0

    def _initialize_position(self):
        # For simplicity, initialize UEs in a 2D space.
        # A more complex model would use a hexagonal grid for eNB positions.
        return np.random.rand(2) * 1000  # Example: 1km x 1km area

    def update_position(self):
        if not self.is_static:
            # Simple random walk for dynamic UEs
            move = (np.random.rand(2) - 0.5) * 2 * MAX_UE_SPEED_MS
            self.position += move

    def receive_data(self, data_mb):
        self.buffered_data_mb += data_mb

class Cell:
    """Represents a single cell within a sector."""
    def __init__(self, cell_type, sector, position):
        self.cell_type = cell_type  # 'A', 'B', 'C', or 'D'
        self.sector = sector
        self.position = position
        self.is_active = True  # Cells start as active
        self.prbs_total = PRB_PER_CELL[self.cell_type]
        self.prbs_used = 0
        self.users = []
        self.power_consumption = 0
        self.deactivation_threshold = 0.2 # Initial value
        self.activation_threshold = 0.8 # Initial value

    @property
    def load(self):
        return self.prbs_used / self.prbs_total if self.prbs_total > 0 else 0

    def update_power_consumption(self):
        if self.is_active:
            p_offset = POWER_OFFSET[self.cell_type]
            p_variable = POWER_VARIABLE[self.cell_type]
            self.power_consumption = p_offset + p_variable * self.load
        else:
            self.power_consumption = POWER_LOW_MODE[self.cell_type]

    def add_user(self, user):
        if user not in self.users:
            self.users.append(user)

    def remove_user(self, user):
        if user in self.users:
            self.users.remove(user)

    def update_status(self, average_load):
        """Updates the cell's on/off status based on load thresholds."""
        if self.is_active and average_load < self.deactivation_threshold:
            self.is_active = False
        elif not self.is_active and average_load > self.activation_threshold:
            self.is_active = True

class Sector:
    """Represents a sector of an eNodeB, containing multiple cells."""
    def __init__(self, sector_id, enb):
        self.sector_id = sector_id
        self.enb = enb
        # For simplicity, cells inherit the eNB's position for now
        self.cells = {cell_type: Cell(cell_type, self, enb.position) for cell_type in CARRIER_BANDS}

    def get_cell(self, cell_type):
        return self.cells.get(cell_type)

class eNodeB:
    """Represents a base station (eNB)."""
    def __init__(self, enb_id, position):
        self.enb_id = enb_id
        self.position = position
        self.sectors = [Sector(i, self) for i in range(SECTORS_PER_ENB)]

    def get_total_power_consumption(self):
        return sum(cell.power_consumption for sector in self.sectors for cell in sector.cells.values())
