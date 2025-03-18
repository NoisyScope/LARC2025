import depthai as dai
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial.transform import Rotation as R

class ExtendedKalmanFilter:
    """ EKF for sensor fusion with IMU data (position, velocity, orientation). """
    def __init__(self, dt, process_noise=1e-3, measurement_noise=1e-2):
        self.dt = dt  # Time step
        
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.x = np.zeros((10, 1))
        self.x[6] = 1  # Initialize quaternion as [1,0,0,0] (identity)

        # State transition matrix (F)
        self.F = np.eye(10)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # Control matrix (maps acceleration to velocity)
        self.B = np.zeros((10, 3))
        self.B[3, 0] = self.dt
        self.B[4, 1] = self.dt
        self.B[5, 2] = self.dt

        # Measurement matrix (accelerometer & gyroscope)
        self.H = np.zeros((6, 10))
        self.H[0:3, 3:6] = np.eye(3)  # Measure velocity (from accelerometer integration)
        self.H[3:6, 6:10] = np.eye(4)  # Measure orientation (from rotation vector)

        # Process covariance (motion model uncertainty)
        self.Q = np.eye(10) * process_noise

        # Measurement covariance (sensor noise)
        self.R = np.eye(6) * measurement_noise

        # Initial state covariance
        self.P = np.eye(10)

    def predict(self, accel):
        """ Predict the next state using motion model. """
        accel = np.array(accel).reshape((3, 1))

        # Update position and velocity
        self.x = np.dot(self.F, self.x) + np.dot(self.B, accel)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, velocity_measurement, quaternion_measurement):
        """ Update state using accelerometer-derived velocity and rotation vector. """
        measurement = np.hstack([velocity_measurement, quaternion_measurement]).reshape((6, 1))
        y = measurement - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.x += np.dot(K, y)  # Update state estimate
        self.P = np.dot(np.eye(10) - np.dot(K, self.H), self.P)  # Update covariance

    def get_position(self):
        return self.x[0:3].flatten()

    def get_velocity(self):
        return self.x[3:6].flatten()

    def get_orientation(self):
        return self.x[6:10].flatten()

# IMU processing functions
def quaternion_to_rotation_matrix(q):
    """Convert quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def remove_gravity(accel, quat):
    """Transform acceleration to world frame and remove gravity."""
    R = quaternion_to_rotation_matrix(quat)
    accel_world = R @ np.array(accel)
    accel_world[2] -= 9.81  # Remove gravity (Z-axis)
    return accel_world

def integrate_gyroscope(gyro, quat, dt):
    """Estimate orientation change using gyroscope data."""
    gyro_vector = np.array([gyro.x, gyro.y, gyro.z]) * dt
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    new_rotation = rotation * R.from_rotvec(gyro_vector)
    return new_rotation.as_quat()


# Import the EKF class (assumed to be defined in a separate file or previous cell)
from ekf_imu_fusion import ExtendedKalmanFilter  

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

# Setup DepthAI pipeline
pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)

# Enable IMU sensors
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)

imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

# Create output queue
xlink_out = pipeline.create(dai.node.XLinkOut)
xlink_out.setStreamName("imu")
imu.out.link(xlink_out.input)

# Initialize EKF
dt = 0.01  # 100 Hz sampling rate
ekf = ExtendedKalmanFilter(dt)

# Run DepthAI device
with dai.Device(pipeline) as device:
    imu_queue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    last_quat = None  # Store last quaternion for gyro integration

    start_time = time.time()

    while True:
        imu_data = imu_queue.get()
        for packet in imu_data.packets:
            # Read Quaternion for orientation
            quat = packet.rotationVector
            q = [quat.real, quat.i, quat.j, quat.k]

            # Read Gyroscope data
            gyro = packet.gyroscope

            # If previous quaternion exists, integrate gyroscope data
            if last_quat is not None:
                q = integrate_gyroscope(gyro, last_quat, dt)
            last_quat = q  # Update stored quaternion

            # Read Accelerometer data
            accel = packet.acceleroMeter.getValues()
            accel_world = remove_gravity([accel.x, accel.y, accel.z], q)

            # EKF Prediction Step
            ekf.predict(accel_world)

            # Estimate velocity using IMU data
            estimated_velocity = ekf.get_velocity()
            ekf.update(estimated_velocity, q)

            # Get Position, Velocity, and Orientation Estimates
            position = ekf.get_position()
            orientation = ekf.get_orientation()

            # Convert quaternion to Euler angles (Yaw, Pitch, Roll)
            euler = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]]).as_euler('zyx', degrees=True)
            
            # Store Data for Plotting
            current_time = time.time() - start_time
            time_data.append(current_time)

            pos_data['x'].append(position[0])
            pos_data['y'].append(position[1])
            pos_data['z'].append(position[2])

            vel_data['vx'].append(estimated_velocity[0])
            vel_data['vy'].append(estimated_velocity[1])
            vel_data['vz'].append(estimated_velocity[2])

            orient_data['yaw'].append(euler[0])
            orient_data['pitch'].append(euler[1])
            orient_data['roll'].append(euler[2])

            # Update Plots
            update_plot()
            
        time.sleep(dt)