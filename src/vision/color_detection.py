import cv2
import numpy as np
from utils.constants import HSV_RANGES

class ColorDetector:
    def detect_colors(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, np.array(lower), np.array(upper)) for lower, upper in HSV_RANGES.values()]
        combined_mask = cv2.bitwise_or.reduce(masks)
        return cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)