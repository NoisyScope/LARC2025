import matplotlib.pyplot as plt
from collections import deque

# Setup real-time plotting
plt.ion()  # Turn on interactive mode

# Buffer size for plotting
BUFFER_SIZE = 200

# Data storage
time_data = deque(maxlen=BUFFER_SIZE)
pos_data = {axis: deque(maxlen=BUFFER_SIZE) for axis in ['x', 'y', 'z']}
vel_data = {axis: deque(maxlen=BUFFER_SIZE) for axis in ['vx', 'vy', 'vz']}
orient_data = {axis: deque(maxlen=BUFFER_SIZE) for axis in ['yaw', 'pitch', 'roll']}

def update_plot():
    """ Updates the real-time plots. """
    plt.clf()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot position
    axs[0].plot(time_data, pos_data['x'], label="X")
    axs[0].plot(time_data, pos_data['y'], label="Y")
    axs[0].plot(time_data, pos_data['z'], label="Z")
    axs[0].set_title("Position (m)")
    axs[0].legend()

    # Plot velocity
    axs[1].plot(time_data, vel_data['vx'], label="Vx")
    axs[1].plot(time_data, vel_data['vy'], label="Vy")
    axs[1].plot(time_data, vel_data['vz'], label="Vz")
    axs[1].set_title("Velocity (m/s)")
    axs[1].legend()

    # Plot orientation (Yaw, Pitch, Roll)
    axs[2].plot(time_data, orient_data['yaw'], label="Yaw")
    axs[2].plot(time_data, orient_data['pitch'], label="Pitch")
    axs[2].plot(time_data, orient_data['roll'], label="Roll")
    axs[2].set_title("Orientation (Degrees)")
    axs[2].legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Pause briefly to update plots