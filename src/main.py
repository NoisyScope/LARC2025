from vision.camera_stream import CameraStream
from vision.color_detection import ColorDetector
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize components
    camera = CameraStream()
    color_detector = ColorDetector()

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
                logging.info("Exiting application")
                break
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        logging.info("Application terminated")

if __name__ == "__main__":
    main()