import os
import json
import numpy as np
from collections import defaultdict


def compute_calibration(raw_results_path: str, output_path: str, alpha: float = 0.1, min_samples: int = 30):
    """
    Validation 결과(raw predictions)를 기반으로 Task 및 Flow-range(bin)별 
    Nonconformity score의 분위수(Quantile)를 계산한다.
    
    alpha = 0.1 이면 90% 오차 구간(Quantile 0.9)을 구함.
    """
    if not os.path.exists(raw_results_path):
        print(f"[!] Validation raw data not found at {raw_results_path}")
        return
        
    with open(raw_results_path, 'r') as f:
        records = json.load(f)
        
    # 기존 "RMSE by True OD Range" 기준과 동일한 bins 사용
    bins = [0, 10, 50, 100, 300, 1000, float('inf')]
    
    # 구조: groups[task][bin_idx] = [scores...]
    groups = defaultdict(lambda: defaultdict(list))
    task_all = defaultdict(list)
    
    eps = 1e-5
    for r in records:
        y_actual = r['y_actual']
        y_pred = r['y_pred']
        task = r['task']
        
        # Heavy-tail 특성을 반영한 상대 오차 (Relative error)
        score = abs(y_actual - y_pred) / (abs(y_actual) + eps)
        
        # Binning (based on True OD for calibration)
        bin_idx = 0
        for i in range(len(bins) - 1):
            if bins[i] <= y_actual < bins[i+1]:
                bin_idx = i
                break
                
        groups[task][bin_idx].append(score)
        task_all[task].append(score)
        
    quantiles = {}
    fallback_count = 0
    
    print(f"\n[Calibration] Computing {(1-alpha)*100:.0f}% quantiles by (task, bin)")
    print(f"{'Task':<5} {'Bin Range':<15} {'Samples':<10} {'Quantile':<10} {'Fallback'}")
    print("-" * 55)
    
    for task in groups.keys():
        # Task 전역 Quantile (Fallback용)
        if len(task_all[task]) > 0:
            task_q = float(np.quantile(task_all[task], 1 - alpha))
        else:
            task_q = 0.5 # Default if completely empty
            
        for i in range(len(bins) - 1):
            key = f"{task}_{i}"
            scores = groups[task][i]
            n_samples = len(scores)
            
            bin_range = f"{bins[i]}~{bins[i+1]}"
            
            if n_samples >= min_samples:
                q = float(np.quantile(scores, 1 - alpha))
                fallback = False
            else:
                q = task_q
                fallback = True
                fallback_count += 1
                
            quantiles[key] = {
                "quantile": q,
                "samples": n_samples,
                "fallback": fallback,
                "bin_range": bin_range
            }
            
            fb_str = "YES (to Task)" if fallback else "NO"
            print(f"{task:<5} {bin_range:<15} {n_samples:<10} {q:<10.4f} {fb_str}")
            
    # 전체 Task 백업도 저장
    for task, scores in task_all.items():
        key = f"{task}_all"
        quantiles[key] = {
            "quantile": float(np.quantile(scores, 1 - alpha)) if scores else 0.5,
            "samples": len(scores),
            "fallback": False,
            "bin_range": "ALL"
        }
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(quantiles, f, indent=2)
        
    print(f"\n[Calibration] Saved to {output_path} (Total Fallbacks: {fallback_count})")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(current_dir, 'validation_raw_results.json')
    out_path = os.path.join(current_dir, 'calibration_quantiles.json')
    
    compute_calibration(raw_path, out_path)
