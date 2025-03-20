import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class CameraStream:
    def __init__(self, camera_index=0):
        """
        Initialize the camera stream.

        Args:
            camera_index (int): The index of the camera to use (default is 0).
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            logging.error(f"Unable to open camera with index {camera_index}")
            raise RuntimeError(f"Unable to open camera with index {camera_index}")
        logging.info(f"Camera with index {camera_index} initialized successfully")

    def get_frame(self):
        """
        Capture a frame from the camera.

        Returns:
            numpy.ndarray: The captured frame in BGR format.
        """
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to capture frame from camera")
            raise RuntimeError("Failed to capture frame from camera")
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera resources released")