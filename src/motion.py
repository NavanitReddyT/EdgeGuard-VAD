import cv2
import numpy as np

def compute_optical_flow(prev_frame, curr_frame):
    """Computes optical flow using the Farneback method."""
    return cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

def compute_frame_difference(prev_frame, curr_frame):
    """Computes the absolute difference between two frames."""
    return cv2.absdiff(prev_frame, curr_frame)

def compute_motion_magnitude(flow):
    """Computes the magnitude of the optical flow."""
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude
