import cv2
import numpy as np

def draw_bounding_boxes(frame, contours, is_anomaly):
    """Draws bounding boxes around contours.

    Args:
        frame: The frame to draw on.
        contours: A list of contours to draw.
        is_anomaly: If True, boxes will be red; otherwise, green.
    """
    color = (0, 0, 255) if is_anomaly else (0, 255, 0) # Red for anomaly, green for normal
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Filter out small boxes to reduce noise
        if cv2.contourArea(contour) > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

def show_anomaly_alert(frame):
    """Adds a red border and text overlay to indicate an anomaly."""
    # Add a red border
    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255])
    # Add text overlay
    text = "ANOMALY DETECTED"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(frame, text, (frame.shape[1] - text_width - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def display_stats(frame, fps, cpu, score):
    """Displays performance stats (HUD) on the frame."""
    hud_text_fps = f"FPS: {fps:.2f}"
    hud_text_cpu = f"CPU: {cpu:.2f}%"
    hud_text_score = f"Score: {score:.4f}"

    cv2.putText(frame, hud_text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, hud_text_cpu, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, hud_text_score, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
