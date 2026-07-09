# config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
DONG_CODE_PATH = os.path.join(BASE_DIR, 'dataset', 'dong_code_cap.xlsx')

# 테스트 지역 행정동 코드
TEST_CITIES_CODES = {
    '동탄': ['3124060', '3124061', '3124062', '3124064', '3124065', '3124066'],
    '위례': ['1124082', '3102168', '3118065'],
    '검단': ['2308069', '2308070', '2308071', '2308076', '2308077']
}

TRAIN_CONFIG = {
    'min_mask_size': 2,
    'max_mask_size': 10,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-3,
    'model_type': 'mae' # gravity, xgb, deep_gravity, mae
}

OUTLIER_CONFIG = {
    # 1. Statistical Outlier Threshold (Z-score > 3 or extreme percentile)
    'statistical_percentile': 99.9,
    # 2. Gravity Model Residual Threshold (Top N%)
    'residual_top_percent': 1.0
}
