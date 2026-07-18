import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mae.train import main as train_mae

if __name__ == '__main__':    
    model_type = input("모델 선택 (1: MAE, 2: twostage): ")
    model_version = input("이전에 테스트 했던 model버전 입력(v1, v2, v3): ")
    
    if (model_type == '1'):
        if (model_version == 'v1'):
            sys.argv = [
                'main.py',
                '--use_friction', 'False',         
                '--od_embed_layers', '1',     
                '--use_self_loop_predictor', 'False',
                '--loss_type', 'weighted_mse',
                '--use_wandb', 'True'          
            ]
        elif (model_version == 'v2'):
            sys.argv = [
                'main.py',
                '--use_friction', 'False',         
                '--od_embed_layers', '2',     
                '--use_self_loop_predictor', 'True',
                '--loss_type', 'weighted_mse',
                '--use_wandb', 'True'          
            ]
        elif (model_version == 'v3'):
            sys.argv = [
                'main.py',
                '--use_friction', 'True',         
                '--od_embed_layers', '3',     
                '--use_self_loop_predictor', 'True',
                '--loss_type', 'weighted_mse',
                '--use_wandb', 'True'          
            ]
    elif (model_type == '2'):
        print("구현중...")
    
    # 학습 스크립트 실행
    train_mae()
