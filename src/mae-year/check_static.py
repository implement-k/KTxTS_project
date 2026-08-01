import pandas as pd

df_19 = pd.read_csv('../dataset/final_static_features_2019.csv')
df_23 = pd.read_csv('../dataset/final_static_features_2023.csv')

cols = ['pop_0_19', 'worker_count', 'business_count']
for c in cols:
    print(f"{c} 2019 mean: {df_19[c].mean():.1f}")
    print(f"{c} 2023 mean: {df_23[c].mean():.1f}")
