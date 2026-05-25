import cv2
import time
import numpy as np
from .pipeline import PipelineManager

def generate_frames(manager: PipelineManager):
    """Generator function for MJPEG streaming."""
    while manager.is_running():
        with manager.lock:
            if manager.current_frame is None:
                continue
            frame = manager.current_frame

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03) # Cap at ~30fps

    # If pipeline stops, show a placeholder
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Pipeline Stopped", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder)
    if flag:
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
