import cv2
import numpy as np
import os
import threading
import time
import logging

from config import ANOMALY_THRESHOLD, DATASET_PATH, MODEL_SAVE_PATH, AUTOENCODER_LATENT_DIM
from src.preprocessor import preprocess_pipeline
from src.features import extract_features
from src.detector import HybridDetector
from src.motion import compute_optical_flow, compute_frame_difference
from src.background import BackgroundSubtractor
from src.benchmarker import Benchmarker
from src.visualizer import draw_bounding_boxes, show_anomaly_alert, display_stats

logging.basicConfig(level=logging.INFO)

class PipelineManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PipelineManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.running = False
        self.mode = "webcam"
        self.threshold = ANOMALY_THRESHOLD
        self.current_frame = None
        self.metrics = {"fps": 0, "cpu": 0, "ram": 0, "score": 0, "anomaly": False, "frame_count": 0}
        self.frame_count = 0
        self.thread = None
        self.lock = threading.Lock()
        self.benchmarker = Benchmarker()
        self.detector = HybridDetector(latent_dim=AUTOENCODER_LATENT_DIM)
        self.detector.load(
            os.path.join(MODEL_SAVE_PATH, "one_class_svm.joblib"),
            os.path.join(MODEL_SAVE_PATH, "autoencoder.pth"),
            AUTOENCODER_LATENT_DIM
        )

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logging.info("Pipeline started.")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            logging.info("Pipeline stopped.")

    def set_mode(self, mode):
        if mode in ["webcam", "dataset"]:
            self.mode = mode
            logging.info(f"Mode switched to {mode}")

    def set_threshold(self, value):
        self.threshold = value
        logging.info(f"Threshold updated to {value}")

    def get_metrics(self):
        return self.metrics

    def is_running(self):
        return self.running

    def _run(self):
        try:
            if self.mode == "webcam":
                cap = cv2.VideoCapture(0)
            else:
                # A bit of a hack to get a test video file. 
                # In a real scenario, you'd have a specific video file path.
                test_video_path = os.path.join(DATASET_PATH, "Test", "Test001.tif") # This is an image, not a video. This will fail.
                # Let's find a video file instead.
                test_video_path = ""
                for root, _, files in os.walk(os.path.join(DATASET_PATH, "Test")):
                    for file in files:
                        if file.endswith( (".avi", ".mp4")):
                            test_video_path = os.path.join(root, file)
                            break
                    if test_video_path:
                        break
                if not test_video_path:
                     raise IOError("No video file found in dataset path")
                cap = cv2.VideoCapture(test_video_path)

            if not cap.isOpened():
                raise IOError(f"Could not open {self.mode}")

            bg_subtractor = BackgroundSubtractor()
            frame_history = []
            prev_frame = None

            while self.running:
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
                    
                    ae_frame = np.array(preprocessed_frame) / 255.0
                    score = self.detector.predict(features, ae_frame)
                    is_anomaly = score > self.threshold

                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    display_frame = draw_bounding_boxes(display_frame, contours, is_anomaly)

                    if is_anomaly:
                        display_frame = show_anomaly_alert(display_frame)

                    self.benchmarker.update()
                    fps = self.benchmarker.get_current_fps()
                    cpu = self.benchmarker.cpu_usages[-1]
                    ram = self.benchmarker.ram_usages[-1]
                    display_frame = display_stats(display_frame, fps, cpu, score)

                    with self.lock:
                        self.current_frame = display_frame
                        self.metrics = {
                            "fps": fps,
                            "cpu": cpu,
                            "ram": ram,
                            "score": score,
                            "anomaly": is_anomaly,
                            "frame_count": self.benchmarker.frame_count
                        }

                    frame_history.append({"fg_mask": fg_mask, "frame_diff": frame_diff})
                    if len(frame_history) > 10:
                        frame_history.pop(0)

                prev_frame = preprocessed_frame
                time.sleep(0.01) # Yield to other threads

        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            # Create an error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Pipeline Error: {e}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            with self.lock:
                self.current_frame = error_frame
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            self.running = False
            logging.info("Pipeline thread finished.")
