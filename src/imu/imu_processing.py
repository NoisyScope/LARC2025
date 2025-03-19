import numpy as np
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