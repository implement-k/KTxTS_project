import os
import sys
import numpy as np
import lightgbm as lgb

# 상위 디렉토리(src) 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import ODDataset

def main():
    print("Dataset 로딩 중...")
    # 학습 환경과 동일하게 Dataset을 로드합니다.
    dataset = ODDataset(mode='train', use_log_transform=True)
    
    print("\nLGBM 모델(대각 성분 전용) 학습 시작...")
    train_idx = dataset.train_indices
    
    # X_static은 이미 numpy array입니다.
    X_train_lgb = dataset.X_static[train_idx]
    y_train_lgb = np.diag(dataset.X_OD)[train_idx]
    
    print(f"학습 데이터 크기: X={X_train_lgb.shape}, y={y_train_lgb.shape}")
    
    lgb_train = lgb.Dataset(X_train_lgb, y_train_lgb)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    print("훈련 진행 중...")
    lgbm_model = lgb.train(params, lgb_train, num_boost_round=100)
    
    # 모델 저장
    best_model_dir = os.path.join(BASE_DIR, '../best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    lgbm_path = os.path.join(best_model_dir, 'best_lgbm_self_loop.txt')
    lgbm_model.save_model(lgbm_path)
    
    print(f"✅ LGBM 모델 저장 완료: {lgbm_path}")
    print("\n이제 다음 명령어로 테스트를 진행하실 수 있습니다:")
    print("python src/mae/model_test.py --use_lgbm_self_loop True --use_mask_channel True")

if __name__ == '__main__':
    main()
