
# simulation/config.py

# --- Simulation Settings (from Table I) ---
ENV_TYPE = "Hex 7"
PACKET_SIZE_MB = 1
SIMULATION_TIME_WEEKS = 1
NUM_CELL_TYPES = 4  # K in the paper
CARRIER_BANDS = ['A', 'B', 'C', 'D']

# Physical Resource Blocks per cell type
PRB_PER_CELL = {
    'A': 200,
    'B': 100,
    'C': 400,
    'D': 200
}

# Transmission power per cell type (in Watts, assumed for now)
P_TX_WATTS = {
    'A': 20, # Example value
    'B': 10,
    'C': 40,
    'D': 20
}

NOISE_POWER_WATTS = 1e-12 # Example value for thermal noise

FIXED_INTERFERENCE_POWER = 1e-11 # Simplified fixed interference power

PATH_LOSS_EXPONENT = 3.0 # Example value for urban environment

PRB_BANDWIDTH_MHZ = 0.18 # Bandwidth of a single PRB in MHz (e.g., 180 kHz)

HEX_GRID_RADIUS_M = 1000 # Radius of the hexagonal grid in meters (e.g., 1 km)



# Power consumption parameters (in Watts)
# Offset power consumption for cell type k
POWER_OFFSET = {
    'A': 117.3,
    'B': 136.83,
    'C': 62.8,
    'D': 62.8
}

# Variable power consumption for cell type k
POWER_VARIABLE = {
    'A': 33.2,
    'B': 50.1,
    'C': 37.3,
    'D': 37.3
}

# Power consumed in low-power mode
POWER_LOW_MODE = {
    'A': 137.0,
    'B': 146.0,
    'C': 119.8,
    'D': 119.8
}

MIN_THROUGHPUT_MBPS = 1.5
MAX_UE_SPEED_MS = 7  # meters per second
SCHEDULER = "Proportional fair scheduler"
DECISION_PERIOD_MIN = 60  # T in the paper (in minutes)
METRIC_GATHERING_FACTOR = 12 # L in the paper

# --- DRL Agent Settings (from Table II) ---
DISCOUNT_FACTOR = 0.99  # gamma
LEARNING_RATE = 1e-3
TARGET_UPDATE_TAU = 5e-3
POLICY_NOISE = 0.2  # sigma
NOISE_CLIP = 0.1 # c
ACTION_MIN = 0.0
ACTION_MAX = 1.0
REPLAY_BUFFER_SIZE = 128 # |B| in the paper

# --- Reward Function Parameters (alpha, beta from Eq. 11) ---
REWARD_ALPHA = 1.0  # Weight for user utility (throughput)
REWARD_BETA = 0.001 # Weight for power consumption

# --- Other settings from the paper ---
NUM_ENB = 7 # B in the paper
SECTORS_PER_ENB = 3
TOTAL_UES_PER_ENB = 55
STATIC_UES_PER_ENB = 15
DYNAMIC_UES_PER_ENB = TOTAL_UES_PER_ENB - STATIC_UES_PER_ENB

# Convert decision period to seconds for simulation
DECISION_PERIOD_SEC = DECISION_PERIOD_MIN * 60
METRIC_GATHERING_PERIOD_SEC = DECISION_PERIOD_SEC / METRIC_GATHERING_FACTOR

# --- Training Settings ---
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = int((SIMULATION_TIME_WEEKS * 7 * 24 * 60) / DECISION_PERIOD_MIN)
