
# Data settings
DATA_FOLDER = './data'
TRAINING_DATA_LIST = './data/train.txt'
TESTING_DATA_LIST = './data/test.txt'
INPUT_D = 56
INPUT_H = 448
INPUT_W = 448
NUM_CLASSES = 2

# HYPERPARAMETERS
RNG_SEED = 1
MODEL_TYPE = 'resnet'
NEW_LAYER_NAMES = ['conv_seg']
MODEL_DEPTH = 50
RESNET_SHORTCUT = 'B' # A = hardcoded downsampling, B = learnable conv + batch norm for downsampling
LEARNING_RATE = 0.001  # Set to 0.001 when finetuning
NUM_EPOCHS = 200

# Model settings
CHECKPOINT_PATH = 'models/resnet_50_epoch_199_batch_0.pth.tar'
PRETRAIN_PATH = 'pretrained_models/resnet_50_23dataset.pth'
SAVE_INTERVALS = 10
SAVE_FOLDER = f"./trails/models/{MODEL_TYPE}_{MODEL_DEPTH}"
PHASE = 'train'

# GPU settings
NO_CUDA = False
GPU_ID = [0]  # List[int], e.g., [0, 1] for multi-GPU or [0] for single GPU
PIN_MEMORY = True if not NO_CUDA else False
NUM_WORKERS = 4
BATCH_SIZE = 1

