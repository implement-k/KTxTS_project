import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DONG_CODE_19_PATH, DONG_CODE_23_PATH,
    STATIC_DATA_19_PATH, STATIC_DATA_23_PATH,
    DATA_DIR
)

def compute_cache_for_year(year):
    print(f"=== Precomputing Merge Cache for {year} ===")
    
    dong_code_path = DONG_CODE_19_PATH if year == '2019' else DONG_CODE_23_PATH
    static_data_path = STATIC_DATA_19_PATH if year == '2019' else STATIC_DATA_23_PATH
    
    # 1. dong 로드
    dong_df = pd.read_excel(dong_code_path)
    dongs = dong_df['dong_code'].astype(int).values
    num_nodes = len(dongs)
    idx2dong = {i: code for i, code in enumerate(dongs)}
    dong2idx = {code: i for i, code in enumerate(dongs)}
    
    # 2. static feature 로드
    static_df = pd.read_csv(static_data_path)
    
    # Rename 2023 specific columns to standard names
    col_mapping = {}
    for c in static_df.columns:
        if c.startswith('station_count_2023_'):
            col_mapping[c] = c.replace('station_count_2023_', 'station_count_')
    static_df.rename(columns=col_mapping, inplace=True)
    
    static_df['dong_code'] = static_df['dong_code'].astype(int)
    static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
    static_df.fillna(0, inplace=True)
    
    # 2.1. static feature density 계산
    static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
    static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
    static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
    
    # 기타지역비율_pct 추가
    static_df['기타지역비율_pct'] = 100.0 - (static_df['상업업무지역비율_pct'] + static_df['공공시설지역비율_pct'] + static_df['주거지역비율_pct'])
    static_df['기타지역비율_pct'] = static_df['기타지역비율_pct'].clip(lower=0.0)
    
    # 2.2. 진짜 feature만 추출 및 indexing
    feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
    raw_static = static_df[feature_cols].values
    
    area_idx = feature_cols.index('행정동전체면적_m2')
    worker_idx = feature_cols.index('worker_count')
    business_idx = feature_cols.index('business_count')
    subway_idx = feature_cols.index('station_count_지하철')
    
    # 3. geojson 로드 및 centroid 계산
    if year == '2019':
        geojson_path = os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20161231.geojson')
    else:
        geojson_path = os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20230101.geojson')
        
    gdf = gpd.read_file(geojson_path)
    
    if year == '2019':
        mapping_path = os.path.join(DATA_DIR, 'preprocessing', 'process', 'mapping_2019_10_to_8.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            c10_to_c8_str = json.load(f)
        c10_to_c8 = {int(k): int(v) for k, v in c10_to_c8_str.items()}
        
        def convert_code(c):
            if pd.isna(c): return 0
            c = int(c)
            if c in c10_to_c8: return c10_to_c8[c]
            return c
            
        gdf['adm_cd'] = pd.to_numeric(gdf['adm_cd'], errors='coerce')
        gdf['adm_cd8'] = gdf['adm_cd'].apply(convert_code).astype(int)
    else:
        manual_dong_mapping = {
            11230740: [11230810],
            31101690: [31101740, 31101750],
            31101700: [31101720, 31101730],
            31103520: [31103620, 31103630],
            31104540: [31104600, 31104610],
            31104590: [31104620, 31104630],
            31250110: [31250600, 31250610, 31250620, 31250630],
        }
        reverse_map = {}
        for old_c, new_cs in manual_dong_mapping.items():
            for nc in new_cs:
                reverse_map[nc] = old_c
                
        def convert_code_23(c):
            if pd.isna(c): return 0
            c = int(c) * 10
            if c in reverse_map: return reverse_map[c]
            return c
            
        gdf['adm_cd'] = pd.to_numeric(gdf['adm_cd'], errors='coerce')
        gdf['adm_cd8'] = gdf['adm_cd'].apply(convert_code_23).astype(int)
        
    gdf = gdf[['adm_cd8', 'geometry']].dissolve(by='adm_cd8').reset_index()
    gdf = gdf[gdf['adm_cd8'].isin(dongs)].copy()
    
    # 3.2. EPSG:5179 (Korea TM)로 투영.
    gdf = gdf.to_crs(epsg=5179)
    gdf['centroid'] = gdf.geometry.centroid
    
    # 3.3. Create centroid array mapped to idx
    centroids_x = np.zeros(num_nodes)
    centroids_y = np.zeros(num_nodes)
    
    missing_codes = []
    missing = 0
    for i in range(num_nodes):
        code = idx2dong[i]
        match = gdf[gdf['adm_cd8'] == code]
        if len(match) > 0:
            centroids_x[i] = match.iloc[0]['centroid'].x
            centroids_y[i] = match.iloc[0]['centroid'].y
        else:
            missing += 1
            missing_codes.append(code)
            centroids_x[i] = np.nan
            centroids_y[i] = np.nan
    
    print(f"Missing geojson matches for {year}: {missing}")
    if missing > 0:
        print(f"Missing dong codes: {missing_codes}")
    
    # Load accurate adjacency we already computed in compute_adj.py
    adj_path = os.path.join(DATA_DIR, 'processed', f'dong_adjacency_{year}.pkl')
    with open(adj_path, 'rb') as f:
        adj_dict = pickle.load(f)
        
    candidates_list = []
    for i in range(num_nodes):
        code_i = idx2dong[i]
        if code_i in adj_dict:
            for code_j in adj_dict[code_i]:
                if code_j in dong2idx:
                    j = dong2idx[code_j]
                    if i < j: # Avoid duplicates (i, j) and (j, i)
                        candidates_list.append((i, j))
                        
    candidates = np.array(candidates_list)
    print(f"Found {len(candidates)} true-adjacency candidate pairs (from adj_dict).")
    
    # 5. Precompute Merge Features
    merge_cache = {}
    
    for idx_a, idx_b in candidates:
        area_a = raw_static[idx_a, area_idx]
        area_b = raw_static[idx_b, area_idx]
        merged_area = area_a + area_b
        
        merged_static = raw_static[idx_a].copy() 
        
        # 1. sum
        count_cols = [c for c in feature_cols if ('pop_' in c) or ('station_count_' in c)]
        for c in count_cols:
            c_idx = feature_cols.index(c)
            merged_static[c_idx] = raw_static[idx_a, c_idx] + raw_static[idx_b, c_idx]
            
        # 2. 면적 override
        merged_static[area_idx] = merged_area
        
        # 3. 종사자수, 사업체 수 면적 비례 배분
        merged_static[worker_idx] = raw_static[idx_a, worker_idx] * (area_a / (merged_area + 1e-5))
        merged_static[business_idx] = raw_static[idx_a, business_idx] * (area_a / (merged_area + 1e-5))
        
        # 4. Area Weighted Average (Percentage variables)
        pct_cols = ['상업업무지역비율_pct', '공공시설지역비율_pct', '주거지역비율_pct', '기타지역비율_pct', '아파트비율_퍼센트']
        for c in pct_cols:
            if c in feature_cols:
                c_idx = feature_cols.index(c)
                val_a = raw_static[idx_a, c_idx]
                val_b = raw_static[idx_b, c_idx]
                merged_static[c_idx] = (val_a * area_a + val_b * area_b) / (merged_area + 1e-5)
            
        # 5. Density (Recalculate)
        worker_den_idx = feature_cols.index('worker_density')
        business_den_idx = feature_cols.index('business_density')
        subway_den_idx = feature_cols.index('station_density_지하철')
        
        merged_static[worker_den_idx] = merged_static[worker_idx] / (merged_area + 1e-5)
        merged_static[business_den_idx] = merged_static[business_idx] / (merged_area + 1e-5)
        merged_static[subway_den_idx] = merged_static[subway_idx] / (merged_area + 1e-5)
        
        # 6. Centroid
        merged_cx = (centroids_x[idx_a] * area_a + centroids_x[idx_b] * area_b) / (merged_area + 1e-5)
        merged_cy = (centroids_y[idx_a] * area_a + centroids_y[idx_b] * area_b) / (merged_area + 1e-5)
        
        # 7. 다른 노드간 거리 재계산
        merged_dist_row = np.sqrt((centroids_x - merged_cx)**2 + (centroids_y - merged_cy)**2) / 1000.0

        merged_self_dist = 0.5 * np.sqrt(merged_area / np.pi) / 1000.0
        merged_dist_row[idx_a] = merged_self_dist
        
        merged_dist_row[idx_b] = np.nan
        
        merge_cache[(idx_a, idx_b)] = {
            'merged_raw_static_at_a': merged_static,
            'merged_dist_row_at_a': merged_dist_row,
            'idx_b_to_deactivate': idx_b
        }
        
    out_path = os.path.join(os.path.dirname(__file__), f'merge_cache_{year}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(merge_cache, f)
        
    print(f"Saved merge_cache_{year} with {len(merge_cache)} pairs to {out_path}\n")

def main():
    compute_cache_for_year('2019')
    compute_cache_for_year('2023')

if __name__ == '__main__':
    main()
