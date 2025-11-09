from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 实验相关配置（用于消融） ====================

# 当前跑的是哪个实验：
#   'baseline'        : 基线模型
#   'no_pe'           : 去掉位置编码
#   'small_dim'       : 缩小模型维度
#   'single_head'     : 单头注意力（无 multi-head）
#   'no_ls'           : 去掉标签平滑
EXPERIMENT_NAME = 'baseline'

# 是否使用位置编码（No Positional Encoding 实验时改为 False）
USE_POSITION_ENCODING = True

# 标签平滑系数（No Label Smoothing 实验时设为 0.0）
LABEL_SMOOTHING = 0.1

# ==================== 模型参数 ====================
DIM_MODEL = 128          # Small Model 实验时可以改小，如 64
NUM_HEADS = 4            # No Multi-Head 时改为 1
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# ==================== 训练参数 ====================
SEQ_LEN = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 30
