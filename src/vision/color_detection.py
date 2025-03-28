import cv2
import numpy as np
from utils.constants import HSV_RANGES
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class ColorDetector:
    def __init__(self, contour_area_threshold=500):
        """
        Initialize the ColorDetector.

        Args:
            contour_area_threshold (int): Minimum area for detected contours.
        """
        self.hsv_ranges = HSV_RANGES
        self.contour_area_threshold = contour_area_threshold

    def detect_colors(self, frame):
        """
        Detect colors in the given frame based on predefined HSV ranges.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.

        Returns:
            dict: A dictionary of detected contours for each color.
        """
        if frame is None:
            logging.error("Invalid frame: None")
            return {}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contours = {}

        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Noise reduction
            detected_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours[color] = detected_contours

            logging.info(f"Detected {len(detected_contours)} contours for color {color}")

        return contours

    def draw_contours(self, frame, contours):
        """
        Draw contours on the frame.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.
            contours (dict): A dictionary of contours for each color.

        Returns:
            numpy.ndarray: The frame with contours drawn.
        """
        if frame is None:
            logging.error("Invalid frame: None")
            return frame

        for color, detected_contours in contours.items():
            for contour in detected_contours:
                if cv2.contourArea(contour) > self.contour_area_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame