"""
配置文件
"""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent


def env_path(name, default):
    return str(Path(os.getenv(name, str(default))).expanduser().resolve())


# 数据路径
EXCEL_PATH = env_path('STYLE_EXCEL_PATH', REPO_ROOT / '室内设计风格.xlsx')
SHEET_NAME = os.getenv('STYLE_IMAGE_SHEET_NAME', '最后喂给AI学习的图片')
STYLE_DEFINITION_SHEET_NAME = os.getenv('STYLE_DEFINITION_SHEET_NAME', '室内风格分类及定义')

# 图片路径
DATA_DIR = env_path('STYLE_DATA_DIR', PROJECT_DIR / 'data')
TRAIN_IMAGE_ROOT = env_path(
    'STYLE_TRAIN_IMAGE_ROOT',
    REPO_ROOT / 'data' / 'images' / 'Interior_style_illustration'
)
PROCESSED_DIR = env_path('STYLE_PROCESSED_DIR', Path(DATA_DIR) / 'processed')

# 模型配置
MODEL_DIR = env_path('STYLE_MODEL_DIR', PROJECT_DIR / 'models')
CHECKPOINT_PATH = env_path('STYLE_CHECKPOINT_PATH', Path(MODEL_DIR) / 'best_model.pth')

# 训练参数
BATCH_SIZE = int(os.getenv('STYLE_BATCH_SIZE', '32'))
LEARNING_RATE = float(os.getenv('STYLE_LEARNING_RATE', '1e-4'))
NUM_EPOCHS = int(os.getenv('STYLE_NUM_EPOCHS', '30'))
EARLY_STOPPING_PATIENCE = int(os.getenv('STYLE_EARLY_STOPPING_PATIENCE', '5'))
VAL_SPLIT = float(os.getenv('STYLE_VAL_SPLIT', '0.2'))
TEST_SPLIT = float(os.getenv('STYLE_TEST_SPLIT', '0.1'))
QUICK_FINE_TUNE_EPOCHS = int(os.getenv('STYLE_QUICK_FINE_TUNE_EPOCHS', '3'))
QUICK_FINE_TUNE_LR = float(os.getenv('STYLE_QUICK_FINE_TUNE_LR', '5e-5'))
QUICK_REPLAY_SAMPLES = int(os.getenv('STYLE_QUICK_REPLAY_SAMPLES', '512'))

# 图片处理
IMAGE_SIZE = int(os.getenv('STYLE_IMAGE_SIZE', '224'))
DOWNLOAD_TIMEOUT = int(os.getenv('STYLE_DOWNLOAD_TIMEOUT', '10'))
TEXT_MAX_FEATURES = int(os.getenv('STYLE_TEXT_MAX_FEATURES', '1024'))
TEXT_EMBED_DIM = int(os.getenv('STYLE_TEXT_EMBED_DIM', '256'))

# 室内风格白名单
# 设为 None 时，默认使用 Excel 中所有满足筛选条件的室内风格。
INTERIOR_STYLE_CATEGORIES = None
EXCLUDED_STYLE_CATEGORIES = [
    '功能主义',
    '奢华商务',
    '性能运动风',
    '机甲风',
    '治愈风',
    '童趣风',
    '越野风',
]

# 模型参数
MODEL_NAME = os.getenv('STYLE_MODEL_NAME', 'resnet50')  # 可选: resnet50, efficientnet_b3

# 设备
DEVICE = os.getenv('STYLE_DEVICE', 'cpu')   # 'cuda' or 'cpu'
NUM_WORKERS = int(os.getenv('STYLE_NUM_WORKERS', '4'))
