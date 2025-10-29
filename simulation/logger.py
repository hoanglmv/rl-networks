
# simulation/logger.py
import os
import datetime

class SimulationLogger:
    """Handles logging of simulation results to a file."""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"simulation_log_{timestamp}.txt")

        with open(self.log_file_path, "w") as f:
            f.write(f"Simulation Log - {timestamp}\n")
            f.write("-------------------------------------\n")

    def log(self, message):
        """Writes a message to the log file and prints to console."""
        with open(self.log_file_path, "a") as f:
            f.write(message + "\n")
        print(message)

    def log_metrics(self, episode, step, reward, avg_power, avg_throughput):
        message = f"Episode: {episode}, Step: {step}, Reward: {reward:.2f}, Avg Power: {avg_power:.2f}W, Avg Throughput: {avg_throughput:.2f}Mbps"
        self.log(message)
