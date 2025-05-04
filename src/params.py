# ========================
# General Data Parameters
# ========================

DATA_ROOT = './data'
S1_PATH = f'{DATA_ROOT}/s1'
S2_PATH = f'{DATA_ROOT}/s2'
MASK_PATH = f'{DATA_ROOT}/s2-mask'

SHUFFLE_DATA = True
DATA_AUGMENTATION = True  # Renamed for consistency

# ============================
# Dataset Splitting Ratios
# ============================

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# ========================
# Data Generator Parameters
# ========================

TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
TESTING_BATCH_SIZE = 16

IMAGE_SIZE = (256, 256)
NUM_CHANNELS = 3
HIDDEN_CHANNELS = 32

# ====================
# Model Parameters
# ====================

DROPOUT_RATE = 0.25
PRETRAINED_WEIGHTS = None

# ====================
# Training Parameters
# ====================

NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # Redundant with TRAINING_BATCH_SIZE; unify if necessary