import pandas as pd

df_19 = pd.read_csv('../dataset/od_data_2019.csv')
df_23 = pd.read_csv('../dataset/od_data_2023.csv')
p = ['귀가', '출근', '등교', '업무', '기타']
print('2019 mean traffic:', df_19[p].sum(axis=1).mean())
print('2023 mean traffic:', df_23[p].sum(axis=1).mean())
