# config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
DIST_DATA_PATH = os.path.join(DATA_DIR, 'dist_data.csv')
STATIC_DATA_PATH = os.path.join(DATA_DIR, 'final_static_features.csv')
OD_DATA_PATH = os.path.join(DATA_DIR, 'od_data.csv')
DONG_CODE_PATH = os.path.join(DATA_DIR, 'raw', 'OD_dong_list.xlsx')

# 테스트용 신도시 행정동 코드
# 2026-07-12 수정: dataset/raw/OD_dong_list.xlsx에 실제 존재하는 dong_code 기준으로 갱신.
# 기존 검단 코드는 현재 OD_dong_list.xlsx에 없어 테스트 구역에 포함되지 않았음.
VAL_CITIES_CODES = {
    '동탄': [
        '31240600',  # 동탄2동
        '31240610',  # 동탄1동
        '31240620',  # 동탄3동
        '31240640',  # 동탄4동
        '31240650',  # 동탄5동
        '31240690',  # 동탄7동
        '31240700',  # 동탄6동
        '31240710',  # 동탄8동
    ],
    '위례': [
        '11240820',  # 위례동
        '31021680',  # 위례동
        '31180650',  # 위례동
    ],
    '검단': [
        '23080800',  # 검단동
        '23080810',  # 불로대곡동
        '23080850',  # 당하동
        '23080860',  # 마전동
        '23080870',  # 원당동
        '23080880',  # 아라동
    ],
}

TEST_CITIES_CODES = {
    '다산': [
        '31130580',  # 다산1동
        '31130590',  # 다산2동
    ],
    '미사': [
        '31180620',  # 미사1동
        '31180630',  # 미사2동
    ],
    '배곧': [
        '31150740',  # 배곧1동
        '31150750',  # 배곧2동
    ],
    '감일동': [
        '31180670',  # 감일동
    ],
    '랜덤1': [
        '11150590', '11040560', '23310320', '23060690', '23040600'
    ],
    '랜덤2': [
        '11140630', '31350360', '11120550'
    ],
    '랜덤3': [
        '31130150', '11050530', '11040700', '11120740', '23020640'
    ],
}

# 마스킹 대상 컬럼
MASKING_COLUMNS = ['worker_count', 'business_count']

TRAIN_CONFIG = {
    'min_mask_size': 1,
    'max_mask_size': 6,
    'batch_size': 32,
    'epochs': 70,
    'learning_rate': 1e-3,
    'model_type': 'mae',
}

OUTLIER_CONFIG = {
    'statistical_percentile': 99.9,
    'residual_top_percent': 1.0
}
