from vision.camera_stream import CameraStream
from vision.color_detection import ColorDetector
from imu.imu_processing import IMUProcessor
from utils.plotting import Plotter

def main():
    # Initialize components
    camera = CameraStream()
    color_detector = ColorDetector()
    imu_processor = IMUProcessor()
    plotter = Plotter()

    while True:
        # Process camera frames
        frame = camera.get_frame()
        detected_objects = color_detector.detect_colors(frame)

        # Process IMU data
        imu_data = imu_processor.get_data()
        imu_processor.update_state(imu_data)

        # Update plots
        plotter.update(imu_processor.get_state(), detected_objects)

if __name__ == "__main__":
    main()