import depthai as dai
import time
from imu_processing import ExtendedKalmanFilter, remove_gravity, integrate_gyroscope
from imu_plotting import update_plot, time_data, pos_data, vel_data, orient_data
from scipy.spatial.transform import Rotation as R

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