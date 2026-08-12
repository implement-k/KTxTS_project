import pandas as pd
import os, torch

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'final_static_features_2023.csv'))

# 1. 컬럼명 리스트만
print(df.columns)
print(df.columns.tolist())   # 리스트 형태로

# 2. 컬럼명 + 개수 + 각 컬럼의 dtype + 결측치 개수까지 한 번에 (제일 유용)
df.info()

# 3. 컬럼별 dtype만 따로
print(df.dtypes)

# 4. "숫자가 아닌(object=문자열 등)" 컬럼만 골라내기 — 지금 에러 원인 찾기에 딱 맞음
non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
print("숫자가 아닌 컬럼:", non_numeric_cols)
