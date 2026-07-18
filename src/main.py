import sys
import os

# src 폴더를 python path에 추가하여 import 에러 방지
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mae.train import main as train_mae

if __name__ == '__main__':
    print("🚀 가장 성능이 좋았던 최신 세팅(거리 저항 O, OD 임베딩 2층)으로 MAE 학습을 시작합니다!")
    print("-------------------------------------------------------------------------")
    
    # 터미널에서 긴 옵션을 칠 필요 없이, 가장 성공적이었던 실험 세팅을 여기에 고정해둡니다.
    sys.argv = [
        'main.py',
        '--use_friction', 'True',         # 중력 모델 (거리 저항) 강제 적용
        '--od_embed_layers', '2',         # 900차원 통행량을 깊게(2층) 압축
        '--loss_type', 'weighted_mse',    # 가장 CPC가 잘 나왔던 순정 Loss
        '--use_wandb', 'False'            # 대시보드 기록 여부 (필요시 True로 변경)
    ]
    
    # 학습 스크립트 실행
    train_mae()
