import json
import os
from typing import Any


def load_calibration_quantiles() -> dict:
    """
    미리 계산된 calibration_quantiles.json을 로드한다.
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'calibration', 'calibration_quantiles.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def get_prediction_uncertainty(
    *,
    predicted_matrix: Any,  # torch.Tensor
    city_codes: list[str],
    od_pairs: list[tuple[str, str]],
    task: int,
    calibration_quantiles: dict,
    bins: list[float] = [0, 10, 50, 100, 300, 1000, float('inf')],
) -> dict[str, dict[str, Any]]:
    """
    특정 통행량(OD pairs)에 대해 Conformal Prediction 기반 
    과거 통계적 오차 범위(Lower/Upper Bound)를 반환한다.
    
    비용: 추가 Forward Pass 없이 Tensor Indexing만 수행 (O(1)에 근접)
    """
    city_codes_str = [str(code) for code in city_codes]
    code_to_idx = {code: idx for idx, code in enumerate(city_codes_str)}
    
    results = {}
    
    for origin_code, dest_code in od_pairs:
        if origin_code not in code_to_idx or dest_code not in code_to_idx:
            continue
            
        o_idx = code_to_idx[origin_code]
        d_idx = code_to_idx[dest_code]
        
        # 1. 예측값 획득
        pred_val = float(predicted_matrix[o_idx, d_idx].item())
        
        # 2. Predicted OD 기준 Bin 판정
        bin_idx = 0
        for i in range(len(bins) - 1):
            if bins[i] <= pred_val < bins[i+1]:
                bin_idx = i
                break
                
        # 3. Quantile 조회
        key = f"{task}_{bin_idx}"
        fallback_key = f"{task}_all"
        
        q_info = None
        if calibration_quantiles:
            if key in calibration_quantiles:
                q_info = calibration_quantiles[key]
            elif fallback_key in calibration_quantiles:
                q_info = calibration_quantiles[fallback_key]
        
        if q_info is None:
            q_info = {"quantile": 0.5, "samples": 0, "fallback": True}
            
        q = q_info["quantile"]
        is_low_confidence = q_info.get("fallback", False)
        sample_size = q_info.get("samples", 0)
        
        # 4. 구간 계산 (Prediction ± q * Prediction)
        # score = |y - y_hat| / |y_hat + eps| (inference 관점)
        lower_bound = max(0.0, pred_val * (1.0 - q))
        upper_bound = pred_val * (1.0 + q)
        
        results[f"{origin_code}->{dest_code}"] = {
            "prediction": pred_val,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": 0.9, # alpha=0.1 하드코딩 기준
            "calibration_sample_size": sample_size,
            "is_low_confidence": is_low_confidence
        }
        
    return results
