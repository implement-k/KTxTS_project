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
TEST_CITIES_CODES = {
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

OUTLIER_CONFIG = {
    'statistical_percentile': 99.9,
    'residual_top_percent': 1.0
}

# ⚠️ 임시 복구 (experiment/loss-comparison 브랜치 한정)
# ff42dd0 커밋("[수정] 중력모델 테스트 검단 행정동 코드 수정")에서 TEST_CITIES_CODES를
# 갱신하는 과정에 아래 두 값이 함께 삭제된 것으로 보여, 삭제 전 값 기준으로 복원함.
# mae1/twostage 학습 파이프라인 실행 복구 목적의 임시 조치이며,
# gravity 담당자 확인 결과에 따라 값이 달라질 수 있음.
MASKING_COLUMNS = ['worker_count', 'business_count']

TRAIN_CONFIG = {
    'min_mask_size': 3,
    'max_mask_size': 10,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-3,
    # 원래 값은 'mae'였으나 train.py의 choices=['lgbm','deep_gravity','mae1','mae5']에 없는 값이라
    # 'mae1'로 수정함(팀 확인 필요 항목).
    'model_type': 'mae1'
}
