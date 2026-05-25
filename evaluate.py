import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from config import DATASET_PATH, MODEL_SAVE_PATH, AUTOENCODER_LATENT_DIM
from src.preprocessor import load_frames, preprocess_pipeline
from src.features import extract_features
from src.detector import HybridDetector
from src.motion import compute_optical_flow, compute_frame_difference
from src.background import BackgroundSubtractor
from src.benchmarker import Benchmarker

def load_ground_truth(gt_folder):
    """Loads ground truth from the UCSD dataset."""
    gt_files = sorted(os.listdir(gt_folder))
    gt = []
    for gt_file in gt_files:
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            # The format is [start_frame:end_frame, ...]
            # For simplicity, we'll just check if the file is empty or not
            # to determine if there are any anomalies in the video.
            # A more robust implementation would parse the frame ranges.
            content = f.read().strip()
            if content:
                gt.append(1) # Anomaly
            else:
                gt.append(0) # Normal
    return gt

def main():
    """Main evaluation script."""
    print("Starting evaluation process...")

    # Load models
    detector = HybridDetector(latent_dim=AUTOENCODER_LATENT_DIM)
    detector.load(
        os.path.join(MODEL_SAVE_PATH, "one_class_svm.joblib"),
        os.path.join(MODEL_SAVE_PATH, "autoencoder.pth"),
        AUTOENCODER_LATENT_DIM
    )
    print("Models loaded.")

    # Load test data and ground truth
    test_path = os.path.join(DATASET_PATH, "Test")
    gt_path = os.path.join(DATASET_PATH, "Test_gt")
    test_folders = sorted([f for f in os.listdir(test_path) if not f.endswith('_gt')])
    ground_truth = load_ground_truth(gt_path)

    benchmarker = Benchmarker()
    all_predictions = []
    all_labels = []

    for i, folder in enumerate(tqdm(test_folders)):
        video_folder = os.path.join(test_path, folder)
        frames = []
        for frame_file in sorted(os.listdir(video_folder)):
            if frame_file.endswith('.tif'):
                frame = load_frames(os.path.join(video_folder, frame_file))[0]
                frames.append(frame)
        
        preprocessed_frames = [preprocess_pipeline(f) for f in frames]

        bg_subtractor = BackgroundSubtractor()
        frame_history = []
        video_scores = []

        for j in range(1, len(preprocessed_frames)):
            prev_frame = preprocessed_frames[j-1]
            curr_frame = preprocessed_frames[j]

            flow = compute_optical_flow(prev_frame, curr_frame)
            fg_mask = bg_subtractor.apply(curr_frame)
            frame_diff = compute_frame_difference(prev_frame, curr_frame)

            features = extract_features(flow, fg_mask, frame_history)
            
            # Normalize frame for autoencoder
            ae_frame = np.array(curr_frame) / 255.0
            score = detector.predict(features, ae_frame)
            video_scores.append(score)

            frame_history.append({"fg_mask": fg_mask, "frame_diff": frame_diff})
            if len(frame_history) > 10:
                frame_history.pop(0)

        # Aggregate scores for the video (e.g., max score)
        final_score = max(video_scores) if video_scores else 0
        prediction = detector.is_anomaly(final_score)
        all_predictions.append(prediction)
        all_labels.append(ground_truth[i])

        benchmarker.update(y_true=ground_truth[i], y_score=final_score)

    # Generate report
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_predictions)}")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_predictions)}")
    benchmarker.generate_report()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    main()
