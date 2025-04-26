import numpy as np
from ahrs.filters import Madgwick
from loguru import logger
import time

class IMUProcessor:
    """
    Process IMU data from OAK-D Pro camera and estimate orientation.
    Uses Madgwick filter for sensor fusion and orientation estimation.
    """
    
    def __init__(self, frequency=100.0):
        """
        Initialize the IMU processor.
        
        Args:
            frequency: IMU data sampling frequency in Hz
        """
        self.frequency = frequency
        self.madgwick = Madgwick(frequency=frequency)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation [w, x, y, z]
        self.last_update_time = None
        logger.info("IMU Processor initialized with frequency: {:.1f} Hz", frequency)
        
    def update(self, gyro_data, accel_data):
        """
        Update orientation estimate using gyroscope and accelerometer data.
        
        Args:
            gyro_data: Gyroscope data [x, y, z] in rad/s
            accel_data: Accelerometer data [x, y, z] in m/s²
            
        Returns:
            Quaternion representing current orientation [w, x, y, z]
        """
        current_time = time.time()
        
        # Convert to numpy arrays
        gyro = np.array(gyro_data)
        accel = np.array(accel_data)
        
        # Calculate actual time delta if not first update
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            # Adjust frequency based on actual time delta
            adjusted_freq = 1.0 / dt if dt > 0 else self.frequency
            self.madgwick.frequency = adjusted_freq
        
        self.last_update_time = current_time
        
        # Update orientation estimate
        self.quaternion = self.madgwick.updateIMU(
            self.quaternion, gyr=gyro, acc=accel
        )
        
        return self.quaternion
    
    def get_euler_angles(self):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Returns:
            Dictionary with roll, pitch, and yaw in degrees
        """
        # Extract quaternion components
        w, x, y, z = self.quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        return {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "yaw": yaw_deg
        }
    
    def get_quaternion(self):
        """
        Get current orientation quaternion.
        
        Returns:
            Quaternion as [w, x, y, z]
        """
        return self.quaternion.tolist()
