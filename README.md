# Real-Time CPU-Based Video Anomaly Detection

This project is a real-time, resource-aware video anomaly detection system built in Python. It uses a hybrid approach of classical computer vision techniques and machine learning to detect anomalies in video streams efficiently on a CPU, without requiring a GPU or cloud infrastructure.

## Features

- **Hybrid Detection:** Combines One-Class SVM for feature-based anomaly detection and a lightweight CNN Autoencoder for reconstruction-based anomaly detection.
- **Real-Time Performance:** Optimized for CPU execution, providing real-time feedback on live video streams.
- **Resource-Aware:** Includes a benchmarker to track FPS, CPU, and RAM usage.
- **Two-Stage Preprocessing:** A deliberate noise filtering pipeline (Median + Gaussian blur) to enhance the performance of background subtraction and optical flow.
- **Modular Design:** The codebase is organized into independent, reusable modules for preprocessing, background subtraction, motion analysis, feature extraction, detection, and visualization.

## Project Structure

```
video-anomaly-detection/
├── data/
│   └── UCSD_Ped2/          # Placeholder for the dataset
├── src/
│   ├── __init__.py
│   ├── preprocessor.py     # Frame preprocessing
│   ├── background.py       # Background subtraction
│   ├── motion.py           # Optical flow and frame differencing
│   ├── features.py         # Spatio-temporal feature extraction
│   ├── detector.py         # Hybrid anomaly detector
│   ├── benchmarker.py      # Performance and accuracy tracking
│   └── visualizer.py       # Visualization tools
├── models/
│   └── autoencoder.py      # Autoencoder model definition
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── demo.py                 # Live webcam demo
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- OpenCV
- NumPy
- Scikit-learn
- PyTorch
- psutil
- Matplotlib
- Seaborn
- joblib

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/video-anomaly-detection.git
   cd video-anomaly-detection
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download the UCSD Ped2 dataset from the official source.
   - Extract the contents into the `data/UCSD_Ped2/` directory.

### Training

To train the models (One-Class SVM and Autoencoder) on the normal frames from the UCSD Ped2 training set, run:

```bash
python train.py
```

This will save the trained models to the `models/` directory.

### Evaluation

To evaluate the trained models on the UCSD Ped2 test set, run:

```bash
python evaluate.py
```

This will print the accuracy, confusion matrix, and a benchmark report, and save an ROC curve plot to the `results/` directory.

### Live Demo

To run the live anomaly detection demo using your webcam, execute:

```bash
python demo.py
```

Press `q` to quit the demo.

## Running the Full Stack App

# Step 1: Train models first (if not done)
python train.py

# Step 2: Start the FastAPI backend
python run.py

# Step 3: Open the frontend
Open frontend/index.html in your browser
(or serve it: python -m http.server 3000 inside frontend/)

# Step 4: In the browser
- Click Start
- Select mode (Dataset or Webcam)
- Adjust threshold slider
- Watch live anomaly detection

API docs auto-available at: http://localhost:8000/docs
