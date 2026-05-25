import cv2
import numpy as np
import os

from config import MODEL_SAVE_PATH, AUTOENCODER_LATENT_DIM
from src.preprocessor import preprocess_pipeline
from src.features import extract_features
from src.detector import HybridDetector
from src.motion import compute_optical_flow, compute_frame_difference
from src.background import BackgroundSubtractor
from src.visualizer import draw_bounding_boxes, show_anomaly_alert, display_stats
from src.benchmarker import Benchmarker

def main():
    """Main demo script for live webcam anomaly detection."""
    print("Starting live webcam demo...")

    # Load models
    detector = HybridDetector(latent_dim=AUTOENCODER_LATENT_DIM)
    detector.load(
        os.path.join(MODEL_SAVE_PATH, "one_class_svm.joblib"),
        os.path.join(MODEL_SAVE_PATH, "autoencoder.pth"),
        AUTOENCODER_LATENT_DIM
    )
    print("Models loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    bg_subtractor = BackgroundSubtractor()
    benchmarker = Benchmarker()
    frame_history = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        preprocessed_frame = preprocess_pipeline(frame)

        if prev_frame is not None:
            flow = compute_optical_flow(prev_frame, preprocessed_frame)
            fg_mask = bg_subtractor.apply(preprocessed_frame)
            frame_diff = compute_frame_difference(prev_frame, preprocessed_frame)

            features = extract_features(flow, fg_mask, frame_history)
            
            # Normalize frame for autoencoder
            ae_frame = np.array(preprocessed_frame) / 255.0
            score = detector.predict(features, ae_frame)
            is_anomaly = detector.is_anomaly(score)

            # Find contours for bounding boxes
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display_frame = draw_bounding_boxes(display_frame, contours, is_anomaly)

            if is_anomaly:
                display_frame = show_anomaly_alert(display_frame)

            # Update stats
            benchmarker.update()
            fps = benchmarker.get_current_fps()
            cpu = benchmarker.cpu_usages[-1]
            display_frame = display_stats(display_frame, fps, cpu, score)

            frame_history.append({"fg_mask": fg_mask, "frame_diff": frame_diff})
            if len(frame_history) > 10:
                frame_history.pop(0)

        cv2.imshow("Anomaly Detection", display_frame)

        prev_frame = preprocessed_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
