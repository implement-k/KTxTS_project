import pickle
import numpy as np
import os, sys

sys.path.append(os.path.abspath(__file__))
with open(os.path.join(os.path.dirname(__file__), 'merge_cache_2019.pkl'), 'rb') as f:
    cache = pickle.load(f)

worker_idx = None  # feature_cols.index('worker_count')와 동일한 값으로 설정
for key, val in cache.items():
    ms = np.asarray(val['merged_raw_static_at_a'], dtype=float)
    if ms.max() > 1e6:
        print(f"극단값 발견: {key}, max={ms.max()}, worker_count 위치 값={ms[worker_idx]}")