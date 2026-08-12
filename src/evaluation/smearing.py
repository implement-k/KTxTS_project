import numpy as np

def compute_smearing_factor(y_true_raw, pred_log):
    """
    y_true_raw: 원본 스케일의 정답값 (0 이상)
    pred_log: 로그 스케일의 예측값 (모델 출력)
    """
    valid_idx = (y_true_raw > 0)
    if not np.any(valid_idx):
        return 1.0
        
    y_true_log = np.log1p(y_true_raw[valid_idx])
    pred_log_valid = pred_log[valid_idx]
    
    residual = y_true_log - pred_log_valid
    
    # Duan's smearing factor
    smearing_factor = np.mean(np.exp(residual))
    
    # 너무 크거나 작은 값 방어
    smearing_factor = np.clip(smearing_factor, 0.5, 2.0)
    
    return float(smearing_factor)

def apply_smearing(T_pred, smearing_factor):
    """
    T_pred: np.expm1(pred_log) 된 예측값 행렬
    """
    return T_pred * smearing_factor
