import cv2
import numpy as np
from config import FRAME_RESIZE_TARGET, MEDIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE

def load_frames(video_path):
    """Extracts frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def resize_frame(frame, size=FRAME_RESIZE_TARGET):
    """Resizes a single frame."""
    return cv2.resize(frame, size)

def convert_grayscale(frame):
    """Converts a frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_noise_filter(frame):
    """
    Applies a two-stage noise filtering pipeline.
    
    Rationale:
    - Stage 1: Median Blur (kernel=3)
      Removes salt-and-pepper/impulse noise from webcam or compression artifacts.
      A small kernel preserves detail while eliminating random pixel noise.
    - Stage 2: Gaussian Blur (kernel=(5,5))
      Smooths intensity gradients, which is crucial for stable Farneback optical flow.
      This is applied after the median filter to work on an already-cleaned signal.
    """
    frame = cv2.medianBlur(frame, MEDIAN_KERNEL_SIZE)
    frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, 0)
    return frame

def preprocess_pipeline(frame):
    """Chains all preprocessing steps together."""
    frame = convert_grayscale(frame)
    frame = resize_frame(frame)
    frame = apply_noise_filter(frame)
    return frame
