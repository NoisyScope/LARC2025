import depthai as dai
import time
import threading
import json
from flask import Flask, render_template, Response
from loguru import logger
import sys

from imu_processor import IMUProcessor

# Configure logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("app.log", rotation="10 MB")

# Initialize Flask app
app = Flask(__name__)

# Global variables
imu_data_lock = threading.Lock()
latest_imu_data = {
    "quaternion": [1.0, 0.0, 0.0, 0.0],
    "euler": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "gyroscope": [0.0, 0.0, 0.0],
    "accelerometer": [0.0, 0.0, 0.0],
    "timestamp": time.time()
}

def initialize_device():
    """Initialize and configure the OAK-D Pro device."""
    logger.info("Initializing OAK-D Pro device")
    
    # Create pipeline
    pipeline = dai.Pipeline()
    
    # Define IMU node
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)
    xlinkOut.setStreamName("imu")
    
    # Enable all IMU sensors
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)  # 100 Hz
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)      # 100 Hz
    imu.setBatchReportThreshold(1)  # Send data as soon as one packet is ready
    imu.setMaxBatchReports(10)      # Maximum number of packets in a batch
    
    # Link IMU to output
    imu.out.link(xlinkOut.input)
    
    return pipeline

def imu_thread_function():
    """Thread function to continuously read IMU data from the device."""
    global latest_imu_data
    
    # Initialize IMU processor
    imu_processor = IMUProcessor(frequency=100.0)
    
    try:
        # Initialize device
        pipeline = initialize_device()
        
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            logger.success("Connected to OAK-D Pro device")
            
            # Get IMU output queue
            imu_queue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
            
            while True:
                # Get IMU data
                imu_data = imu_queue.get()
                
                # Process accelerometer data
                accel_values = imu_data.acceleroMeter
                accel_x = accel_values.x
                accel_y = accel_values.y
                accel_z = accel_values.z
                accel_timestamp = accel_values.timestamp.total_seconds()
                
                # Process gyroscope data
                gyro_values = imu_data.gyroscope
                gyro_x = gyro_values.x
                gyro_y = gyro_values.y
                gyro_z = gyro_values.z
                gyro_timestamp = gyro_values.timestamp.total_seconds()
                
                # Update orientation estimate
                quaternion = imu_processor.update(
                    gyro_data=[gyro_x, gyro_y, gyro_z],
                    accel_data=[accel_x, accel_y, accel_z]
                )
                
                # Get Euler angles
                euler_angles = imu_processor.get_euler_angles()
                
                # Update latest IMU data with thread safety
                with imu_data_lock:
                    latest_imu_data = {
                        "quaternion": quaternion.tolist(),
                        "euler": euler_angles,
                        "gyroscope": [gyro_x, gyro_y, gyro_z],
                        "accelerometer": [accel_x, accel_y, accel_z],
                        "timestamp": time.time()
                    }
                
                # Log data occasionally (every 5 seconds)
                if int(time.time()) % 5 == 0:
                    logger.debug(f"Orientation: Roll={euler_angles['roll']:.1f}°, "
                                f"Pitch={euler_angles['pitch']:.1f}°, "
                                f"Yaw={euler_angles['yaw']:.1f}°")
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.001)
                
    except Exception as e:
        logger.error(f"Error in IMU thread: {e}")
        # If device not found, keep retrying
        if "device not found" in str(e).lower():
            logger.warning("OAK-D Pro device not found. Retrying in 5 seconds...")
            time.sleep(5)
            imu_thread_function()  # Recursive retry

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/imu-data')
def imu_data():
    """Server-sent events endpoint for IMU data."""
    def generate():
        last_sent_time = 0
        while True:
            current_time = time.time()
            # Send data at most 30 times per second
            if current_time - last_sent_time >= 1/30:
                with imu_data_lock:
                    data = latest_imu_data.copy()
                
                yield f"data: {json.dumps(data)}\n\n"
                last_sent_time = current_time
            
            time.sleep(0.01)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Start IMU thread
    imu_thread = threading.Thread(target=imu_thread_function, daemon=True)
    imu_thread.start()
    
    logger.info("Starting web server on port 5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
