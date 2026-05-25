# Frame processing configuration
FRAME_RESIZE_TARGET = (128, 128)

# Optical flow configuration
OPTICAL_FLOW_METHOD = 'Farneback'

# Background subtraction configuration
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 50

# Noise filtering configuration
MEDIAN_KERNEL_SIZE = 3
GAUSSIAN_KERNEL_SIZE = (5, 5)

# One-Class SVM configuration
OC_SVM_KERNEL = 'rbf'
OC_SVM_NU = 0.1

# Autoencoder configuration
AUTOENCODER_LATENT_DIM = 64
AUTOENCODER_EPOCHS = 30

# Anomaly detection configuration
ANOMALY_THRESHOLD = 0.5

# Paths
DATASET_PATH = 'data/UCSD_Ped2'
MODEL_SAVE_PATH = 'models/'
