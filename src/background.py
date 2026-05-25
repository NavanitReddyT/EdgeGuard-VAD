import cv2
from config import MOG2_HISTORY, MOG2_VAR_THRESHOLD

class BackgroundSubtractor:
    """Wrapper for the MOG2 background subtraction algorithm."""
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False
        )

    def apply(self, frame):
        """Applies background subtraction to a frame."""
        return self.subtractor.apply(frame)

    def get_foreground_region(self, frame, mask):
        """Extracts the foreground region from a frame using a mask."""
        return cv2.bitwise_and(frame, frame, mask=mask)
