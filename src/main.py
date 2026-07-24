import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mae.train import main as train_mae
from twostage.train import main as train_twostage
from evaluation_pipeline import run_evaluation_pipeline

if __name__ == '__main__':    
    print("=== KTDB 모델 학습 및 평가 ===")
    mode = input("모드 선택 (1: 학습, 2: 평가): ")
    if (mode != '1' and mode != '2'):
        print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")
        sys.exit(1)
    
    if mode == '1':
        model_type = input("모델 선택 (1: MAE, 2: twostage): ")
        if (model_type != '1' and model_type != '2'):
            print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")
            sys.exit(1)
    
        if model_type == '1':
            model_version = input("이전에 테스트 했던 model버전 입력(v1, v2, v3): ")
            if model_version == 'v1':
                sys.argv = [
                    'main.py',
                    '--use_friction', 'False',         
                    '--od_embed_layers', '1',     
                    '--use_self_loop_predictor', 'False',
                    '--loss_type', 'weighted_mse',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v2':
                sys.argv = [
                    'main.py',
                    '--use_friction', 'False',         
                    '--od_embed_layers', '2',     
                    '--use_self_loop_predictor', 'True',
                    '--loss_type', 'weighted_mse',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v3':
                sys.argv = [
                    'main.py',
                    '--use_friction', 'False',         
                    '--od_embed_layers', '3',     
                    '--use_self_loop_predictor', 'True',
                    '--loss_type', 'weighted_mse',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v4':
                sys.argv = [
                    'main.py',
                    '--use_friction', 'True',         
                    '--od_embed_layers', '3',     
                    '--use_self_loop_predictor', 'True',
                    '--loss_type', 'weighted_mse',
                    '--use_wandb', 'True'          
                ]
            train_mae()
            
        elif model_type == '2':
            model_version = input("이전에 테스트 했던 model버전 입력(v2, v3, v4): ")
            if model_version == 'v2':
                sys.argv = [
                    'main.py',
                    '--epochs', '40',
                    '--predict_only_masked', 'False',
                    '--use_residual', 'False',
                    '--use_od', 'False',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v3':
                sys.argv = [
                    'main.py',
                    '--epochs', '40',
                    '--use_4_lgbm', 'True',
                    '--use_nan_masking', 'True',
                    '--predict_only_masked', 'False',
                    '--use_residual', 'False',
                    '--use_od', 'False',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v4':
                sys.argv = [
                    'main.py',
                    '--epochs', '40',
                    '--predict_only_masked', 'False',
                    '--use_od', 'True',
                    '--use_wandb', 'True'          
                ]
            elif model_version == 'v5':
                sys.argv = [
                    'main.py',
                    '--epochs', '40',
                    '--predict_only_masked', 'True',
                    '--use_residual', 'True',
                    '--use_od', 'True',
                    '--use_wandb', 'True'          
                ]
            else:
                print("지원하지 않는 버전입니다.")
                
            train_twostage()
    elif mode == '2':
        
        run_evaluation_pipeline()
    
    
