import os
import numpy as np
from tqdm import tqdm

from config import DATASET_PATH, MODEL_SAVE_PATH, AUTOENCODER_LATENT_DIM
from src.preprocessor import load_frames, preprocess_pipeline
from src.features import extract_features
from src.detector import OneClassSVMDetector, AutoencoderDetector
from src.motion import compute_optical_flow, compute_frame_difference
from src.background import BackgroundSubtractor

def main():
    """Main training script."""
    print("Starting training process...")

    # Load normal training frames from UCSD Ped2
    train_path = os.path.join(DATASET_PATH, "Train")
    normal_frames = []
    for folder in sorted(os.listdir(train_path)):
        if "_gt" not in folder:
            video_folder = os.path.join(train_path, folder)
            for video_file in sorted(os.listdir(video_folder)):
                if video_file.endswith('.tif'): # Assuming .tif files are frames
                    frame_path = os.path.join(video_folder, video_file)
                    frame = load_frames(frame_path)[0] # load_frames returns a list
                    normal_frames.append(frame)

    print(f"Loaded {len(normal_frames)} normal frames.")

    # Preprocess all frames
    print("Preprocessing frames...")
    preprocessed_frames = [preprocess_pipeline(frame) for frame in tqdm(normal_frames)]

    # Feature extraction
    print("Extracting features...")
    feature_vectors = []
    bg_subtractor = BackgroundSubtractor()
    frame_history = []
    for i in tqdm(range(1, len(preprocessed_frames))):
        prev_frame = preprocessed_frames[i-1]
        curr_frame = preprocessed_frames[i]

        flow = compute_optical_flow(prev_frame, curr_frame)
        fg_mask = bg_subtractor.apply(curr_frame)
        frame_diff = compute_frame_difference(prev_frame, curr_frame)

        features = extract_features(flow, fg_mask, frame_history)
        feature_vectors.append(features)

        frame_history.append({"fg_mask": fg_mask, "frame_diff": frame_diff})
        if len(frame_history) > 10: # Keep history limited
            frame_history.pop(0)

    # Train One-Class SVM
    print("Training One-Class SVM...")
    svm_detector = OneClassSVMDetector()
    svm_detector.fit(feature_vectors)
    svm_detector.save(os.path.join(MODEL_SAVE_PATH, "one_class_svm.joblib"))
    print("One-Class SVM training complete.")

    # Train Autoencoder
    print("Training Autoencoder...")
    ae_detector = AutoencoderDetector(latent_dim=AUTOENCODER_LATENT_DIM)
    # Normalize frames for autoencoder
    ae_frames = np.array(preprocessed_frames) / 255.0
    ae_detector.fit(ae_frames)
    ae_detector.save(os.path.join(MODEL_SAVE_PATH, "autoencoder.pth"))
    print("Autoencoder training complete.")

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    main()
