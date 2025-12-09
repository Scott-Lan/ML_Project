# name: Scott Landry
# Configuration file for video classification project
# All hyperparameters and paths are defined here as a single source of truth

# Dataset paths
ROOT = "./deepaction_videos"  # root directory containing video class folders
DATASET_PATH = './deepaction_videos'  # alias for compatibility with deepaction_extract.py

# Data splitting
TRAIN_SIZE = 0.8  # proportion of data for training
VAL_SIZE = 0.1    # validation
TEST_SIZE = 0.1   # testing

# Image preprocessing
IMG_SIZE = 256  # image resolution (height and width in pixels)

# Video sampling
NUM_FRAMES = 20  # number of frames to sample per video

# Training hyperparameters
BATCH_SIZE = 16  # batch size - smaller batches often improve accuracy (8 is good) (16 starts getting hungery for memory - 12gb vram)
LEARNING_RATE = 0.0008  # learning rate - lower more stable, higher is faster
NUM_EPOCHS = 10  # number of training epochs - more epochs can improve accuracy ------ 10 works well but long

# Reproducibility
#RANDOM_SEED = 42  # random seed for reproducibility

