import cv2

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
            raise RuntimeError(f"Unable to open camera with index {camera_index}")

    def get_frame(self):
        """
        Capture a frame from the camera.

        Returns:
            numpy.ndarray: The captured frame in BGR format.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()