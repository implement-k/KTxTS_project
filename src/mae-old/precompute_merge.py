import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DONG_CODE_PATH, STATIC_DATA_PATH, DATA_DIR
)

def main():
    # 1. dong 로드
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dongs = dong_df['dong_code'].astype(int).values
    num_nodes = len(dongs)
    idx2dong = {i: code for i, code in enumerate(dongs)}
    dong2idx = {code: i for i, code in enumerate(dongs)}
    
    # 2. static feature 로드
    static_df = pd.read_csv(STATIC_DATA_PATH)
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
    gdf = gpd.read_file(os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20220101.geojson'))
    gdf['adm_cd8'] = gdf['adm_cd8'].astype(int)
    
    # 3.1. dong 기준으로 필터링
    gdf = gdf[gdf['adm_cd8'].isin(dongs)].copy()
    
    # 3.2. EPSG:5179 (Korea TM)로 투영.
    gdf = gdf.to_crs(epsg=5179)
    gdf['centroid'] = gdf.geometry.centroid
    
    # 3.3. Create centroid array mapped to idx
    centroids_x = np.zeros(num_nodes)
    centroids_y = np.zeros(num_nodes)
    
    # 3.4. missing dong code 확인(여기서 없는 경우는 수도권이 아니므로 괜찮음, 모두 매칭되는 것 이전에 확인 함.)
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
    
    print(f"Missing geojson matches: {missing}")
    if missing > 0:
        print(f"Missing dong codes: {missing_codes}")
    
    # 4. 인접 동 추출
    gdf_indexed = gdf.copy()
    gdf_indexed['node_idx'] = gdf_indexed['adm_cd8'].map(dong2idx)
    gdf_indexed = gdf_indexed.dropna(subset=['node_idx']).reset_index(drop=True)
    gdf_indexed['node_idx'] = gdf_indexed['node_idx'].astype(int)
    
    sindex = gdf_indexed.sindex
    candidates_list = []
    
    for i, row in gdf_indexed.iterrows():
        geom = row.geometry
        idx_i = row['node_idx']
        
        possible_matches_index = list(sindex.intersection(geom.bounds))
        
        for j in possible_matches_index:
            if i == j:
                continue
            idx_j = gdf_indexed.iloc[j]['node_idx']
            other_geom = gdf_indexed.iloc[j].geometry
            
            # 실제 경계를 공유하는지 (점 하나만 맞닿는 경우도 포함)
            if geom.touches(other_geom):
                candidates_list.append((idx_i, idx_j))
                
    candidates = np.array(candidates_list)
    print(f"Found {len(candidates)} true-adjacency candidate pairs (shares border).")
    
    # 5. Precompute Merge Features
    merge_cache = {}
    
    for idx_a, idx_b in candidates:
        area_a = raw_static[idx_a, area_idx]
        area_b = raw_static[idx_b, area_idx]
        merged_area = area_a + area_b
        
        # a는 아는 노드, b는 마스킹된 노드라고 가정. 따라서 worker count는 area_a의 worker count에 비례.
        merged_static = raw_static[idx_a].copy() 
        
        # 1. sum
        count_cols = [c for c in feature_cols if ('pop_' in c) or ('station_count_' in c)]
        for c in count_cols:
            c_idx = feature_cols.index(c)
            merged_static[c_idx] = raw_static[idx_a, c_idx] + raw_static[idx_b, c_idx]
            
        # 2. 면적 override
        merged_static[area_idx] = merged_area
        
        # 3. 종사자수, 사업체 수 면적 비례 배분 (당연히 부정확함. indicator로 표시하여 모델이 학습 할 수 있도록 해야 함)
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
        
    # Verification of unhandled features
    handled = set(count_cols) | {'worker_count', 'business_count', '행정동전체면적_m2'} | set(pct_cols) | {'worker_density', 'business_density', 'station_density_지하철'}
    unhandled = [c for c in feature_cols if c not in handled]
    if len(unhandled) > 0:
        print(f"WARNING: 병합 로직에서 명시적으로 처리되지 않은 컬럼이 있습니다: {unhandled}")
    else:
        print("모든 feature 컬럼이 병합 로직에 의해 명시적으로 안전하게 처리되었습니다.")

    out_path = os.path.join(os.path.dirname(__file__), 'merge_cache.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(merge_cache, f)
        
    print(f"Saved merge_cache with {len(merge_cache)} pairs to {out_path}")

if __name__ == '__main__':
    main()
