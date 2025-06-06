# Data paths
DATA_FOLDER = './data'
TRAINING_DATA_LIST = './data/train.txt'
TESTING_DATA_LIST = './data/test.txt'

# Data settings
INPUT_D = 56
INPUT_H = 448
INPUT_W = 448

# Model settings
CHECKPOINT_PATH = ''
PRETRAIN_PATH = 'pretrain/resnet_50.pth'
NEW_LAYER_NAMES = ['conv_seg']
MODEL_TYPE = 'resnet'
MODEL_DEPTH = 50
RESNET_SHORTCUT = 'B'
SAVE_FOLDER = f"./trails/models/{MODEL_TYPE}_{MODEL_DEPTH}"

# GPU settings
NO_CUDA = False
GPU_ID = None  # List[int], e.g., [0, 1]
PIN_MEMORY = True if not NO_CUDA else False

# Training settings
NUM_CLASSES = 2
LEARNING_RATE = 0.001  # Set to 0.001 when finetuning
NUM_WORKERS = 4
BATCH_SIZE = 1
PHASE = 'train'
SAVE_INTERVALS = 10
NUM_EPOCHS = 200
RNG_SEED = 1
