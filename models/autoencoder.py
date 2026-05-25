import torch
import torch.nn as nn
from config import FRAME_RESIZE_TARGET

class TinyAutoencoder(nn.Module):
    """A tiny CNN autoencoder for anomaly detection."""
    def __init__(self, latent_dim):
        super(TinyAutoencoder, self).__init__()
        input_size = FRAME_RESIZE_TARGET[0] * FRAME_RESIZE_TARGET[1]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            nn.Sigmoid() # Use Sigmoid for pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
