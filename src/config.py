# config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
DIST_DATA_PATH = os.path.join(DATA_DIR, 'dist_data.csv')
STATIC_DATA_PATH = os.path.join(DATA_DIR, 'final_static_features.csv')
OD_DATA_PATH = os.path.join(DATA_DIR, 'od_data.csv')
DONG_CODE_PATH = os.path.join(DATA_DIR, 'raw', 'OD_dong_list.xlsx')

# 테스트용 신도시 행정동 코드
TEST_CITIES_CODES = {
    '동탄': ['31240600', '31240610', '31240620', '31240640', '31240650', '31240660'],
    '위례': ['11240820', '31021680', '31180650'],
    '검단': ['23080690', '23080700', '23080710', '23080760', '23080770']
}

# 마스킹 대상 컬럼 (신도시 예측때 masking)
MASKING_COLUMNS = ['worker_count', 'business_count']

TRAIN_CONFIG = {
    'min_mask_size': 2,
    'max_mask_size': 10,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-3,
    'model_type': 'mae' 
}

OUTLIER_CONFIG = {
    'statistical_percentile': 99.9,
    'residual_top_percent': 1.0
}
