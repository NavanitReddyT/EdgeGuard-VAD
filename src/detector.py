import joblib
import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from config import OC_SVM_KERNEL, OC_SVM_NU, ANOMALY_THRESHOLD, AUTOENCODER_EPOCHS
from models.autoencoder import TinyAutoencoder

class OneClassSVMDetector:
    """Wrapper for the One-Class SVM detector."""
    def __init__(self):
        self.model = OneClassSVM(kernel=OC_SVM_KERNEL, nu=OC_SVM_NU)

    def fit(self, X):
        """Train the One-Class SVM model."""
        self.model.fit(X)

    def score(self, x):
        """Get the anomaly score for a feature vector."""
        return self.model.decision_function([x])[0]

    def save(self, path):
        """Save the model to a file."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load the model from a file."""
        self.model = joblib.load(path)

class AutoencoderDetector:
    """Wrapper for the Autoencoder detector."""
    def __init__(self, latent_dim):
        self.model = TinyAutoencoder(latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, frames):
        """Train the Autoencoder model."""
        dataset = TensorDataset(torch.Tensor(frames))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(AUTOENCODER_EPOCHS):
            for data in loader:
                img, = data
                img = img.view(img.size(0), -1)
                recon = self.model(img)
                loss = self.criterion(recon, img)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def score(self, frame):
        """Get the reconstruction error for a frame."""
        frame_tensor = torch.Tensor(frame).view(1, -1)
        recon = self.model(frame_tensor)
        return self.criterion(recon, frame_tensor).item()

    def save(self, path):
        """Save the model state dict."""
        torch.save(self.model.state_dict(), path)

    def load(self, path, latent_dim):
        """Load the model state dict."""
        self.model = TinyAutoencoder(latent_dim)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class HybridDetector:
    """Combines One-Class SVM and Autoencoder detectors."""
    def __init__(self, latent_dim):
        self.svm_detector = OneClassSVMDetector()
        self.ae_detector = AutoencoderDetector(latent_dim)

    def predict(self, features, frame):
        """Get a combined anomaly score."""
        svm_score = self.svm_detector.score(features)
        ae_score = self.ae_detector.score(frame)
        # Simple weighted average, can be tuned
        return 0.5 * svm_score + 0.5 * ae_score

    def is_anomaly(self, score):
        """Check if a score indicates an anomaly."""
        return score > ANOMALY_THRESHOLD

    def save(self, svm_path, ae_path):
        """Save both models."""
        self.svm_detector.save(svm_path)
        self.ae_detector.save(ae_path)

    def load(self, svm_path, ae_path, latent_dim):
        """Load both models."""
        self.svm_detector.load(svm_path)
        self.ae_detector.load(ae_path, latent_dim)
