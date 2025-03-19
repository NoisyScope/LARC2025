from vision.camera_stream import CameraStream
from vision.color_detection import ColorDetector
from imu.imu_processing import IMUProcessor
from utils.plotting import Plotter
import cv2

def main():
    # Initialize components
    camera = CameraStream()
    color_detector = ColorDetector()
    imu_processor = IMUProcessor()
    plotter = Plotter()

    try:
        while True:
            # Capture a frame
            frame = camera.get_frame()

            # Detect colors
            contours = color_detector.detect_colors(frame)

            # Draw contours on the frame
            frame_with_contours = color_detector.draw_contours(frame, contours)

            # Display the frame
            cv2.imshow("Color Detection", frame_with_contours)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Process IMU data
            imu_data = imu_processor.get_data()
            imu_processor.update_state(imu_data)

            # Update plots
            plotter.update(imu_processor.get_state(), contours)
    finally:
        # Release resources
        camera.release()

if __name__ == "__main__":
    main()