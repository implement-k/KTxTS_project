import pickle
import pandas as pd
import os

year = '2023'
df = pd.read_excel(f'/Users/implement/KT/KTDB/dataset/raw/dong/OD_dong_list_{year}.xlsx')
if 'dong_name_orig' in df.columns and 'dong_name' not in df.columns:
    df.rename(columns={'dong_name_orig': 'dong_name'}, inplace=True)
cols = [c for c in ['dong_code', 'dong_name'] if c in df.columns]
df_unique = df[cols].drop_duplicates(subset=['dong_code'], keep='first')
df_unique.to_excel(f'/Users/implement/KT/KTDB/dataset/raw/dong/OD_dong_list_{year}_unique.xlsx', index=False)
print(f'2023년 unique xlsx 재생성 완료: {len(df_unique)}행')

def check_coverage(year):
    adj_path = f'/Users/implement/KT/KTDB/dataset/processed/dong_adjacency_{year}.pkl'
    od_path = f'/Users/implement/KT/KTDB/dataset/raw/dong/OD_dong_list_{year}_unique.xlsx'
    
    with open(adj_path, 'rb') as f:
        adj_dict = pickle.load(f)
        
    od_df = pd.read_excel(od_path)
    target_codes = set(od_df['dong_code'].astype(int))
    adj_codes = set(adj_dict.keys())
    
    missing_from_adj = target_codes - adj_codes
    
    print(f"--- {year}년 ---")
    print(f"Target 노드 개수: {len(target_codes)}")
    print(f"Adjacency 행렬 내 존재하는 타겟 노드: {len(target_codes.intersection(adj_codes))}")
    print(f"누락된 타겟 노드 개수: {len(missing_from_adj)}")
    
    if missing_from_adj:
        missing_df = od_df[od_df['dong_code'].isin(missing_from_adj)]
        print("누락된 동 리스트:")
        for _, row in missing_df.iterrows():
            print(f" - {row['dong_code']}: {row['dong_name']}")
    print("")

check_coverage('2019')
check_coverage('2023')
