import csv
import os
from utils import current_timestamp

LOG_FILE = "vehicle_log.csv"

def init_log():
    """Initializes the log file if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Direction", "Vehicle Type", "Total Count"])

def log_event(direction, vehicle_type, count):
    """Logs a counting event."""
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_timestamp(), direction, vehicle_type, count])
