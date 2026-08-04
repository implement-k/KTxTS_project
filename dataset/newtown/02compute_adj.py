import os, pickle
import geopandas as gpd
import pandas as pd
import json

def compute_adj(geo_path):
    # 파일 열기    
    gdf = gpd.read_file(geo_path)
    
    # geojson의 코드가 7자리인 경우 끝에 '0'을 붙여 8자리로 보정
    gdf['ADM_CD'] = gdf['ADM_CD'].apply(lambda x: str(x).strip() + '0' if len(str(x).strip()) == 7 else str(x))
    gdf['ADM_CD'] = gdf['ADM_CD'].astype(int)

    print('Projecting...')
    gdf = gdf.to_crs(epsg=5179)

    print('Calculating adjacency...')
    sindex = gdf.sindex
    adj_dict = {}

    for i, row in gdf.iterrows():
        code_i = row['ADM_CD']
        geom = row.geometry
        possible_matches_index = list(sindex.intersection(geom.bounds))
        
        adj = []
        for j in possible_matches_index:
            if i == j:
                continue
            other_geom = gdf.iloc[j].geometry
            # Check touches
            if geom.touches(other_geom):
                adj.append(int(gdf.iloc[j]['ADM_CD']))
                
        if len(adj) == 0:
            print(f"  [Warning] 실패동 (이웃 없음): {code_i}")
            
        adj_dict[code_i] = adj


    manual_dong_mapping = {
        11230740: [11230810],
        31101690: [31101740, 31101750],
        31101700: [31101720, 31101730],
        31103520: [31103620, 31103630],
        31104540: [31104600, 31104610],
        31104590: [31104620, 31104630],
        31250110: [31250600, 31250610, 31250620, 31250630],
    }
    for old_code, new_codes in manual_dong_mapping.items():
        if old_code in adj_dict:
            old_neighbors = adj_dict[old_code]
            del adj_dict[old_code]
            
            for new_code in new_codes:
                adj_dict[new_code] = list(old_neighbors)
                for other_new_code in new_codes:
                    if new_code != other_new_code:
                        adj_dict[new_code].append(other_new_code)
                        
            for n in old_neighbors:
                if n in adj_dict:
                    if old_code in adj_dict[n]:
                        adj_dict[n].remove(old_code)
                        adj_dict[n].extend(new_codes)
        else:
            print(f"  [Warning] 매핑 실패동 (adj_dict에 없음): {old_code}")
                            
        for k in adj_dict:
            adj_dict[k] = list(set(adj_dict[k]))

    # 수동 이웃 추가 (5로 시작하는 코드들)
    manual_adj = {
        50000001: [3118053, 3118052, 3118054, 3118051, 50000001],
        50000002: [3118054, 50000002, 3118051, 3118059, 50000003], 
        50000003: [3118059, 50000004],
        50000004: [50000003, 3118059, 3118067],
        50000005: [3113012, 50000006, 3113014],
        50000006: [3113012, 50000005, 3113016, 3113014, 50000007],
        50000007: [3113016, 3113058, 3112052, 3113014, 50000006],
        50000008: [3113014],
        50000009: [3113058, 3113054],
        50000010: [3110153, 3110163, 3110169, 50000011],
        50000011: [50000010, 3110169, 3110153, 3110166, 50000013, 3110168, 50000012],
        50000012: [50000011, 50000013, 3110158, 3110153, 3110170],
        50000013: [50000012, 50000011, 3110153, 3110167, 3110158, 3110168, 3110166],
        50000014: [3110168, 3110166]
    }
    
    for k, neighbors in manual_adj.items():
        fixed_k = k
        if len(str(fixed_k)) == 7:
            fixed_k = int(str(fixed_k) + '0')
            
        for n in neighbors:
            fixed_n = n
            if len(str(fixed_n)) == 7:
                fixed_n = int(str(fixed_n) + '0')
                
            if fixed_n == fixed_k:
                continue
                
            if fixed_k not in adj_dict:
                adj_dict[fixed_k] = []
            if fixed_n not in adj_dict[fixed_k]:
                adj_dict[fixed_k].append(fixed_n)
                
            if fixed_n not in adj_dict:
                adj_dict[fixed_n] = []
            if fixed_k not in adj_dict[fixed_n]:
                adj_dict[fixed_n].append(fixed_k)

    return adj_dict

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = ['all', 'changneung', 'wangsuk', 'gyosan']
    
    for folder in folders:
        print(f"========== Checking folder: {folder} ==========")
        geo_path = os.path.join(base_dir, folder, 'dong_area.geojson')

        if not os.path.exists(geo_path):
            print(f"E: {geo_path} 파일이 없습니다.")
            continue
            
        adj_dict = compute_adj(geo_path)
        out_path = os.path.join(base_dir, folder, f'dong_adjacency.pkl')
        
        with open(out_path, 'wb') as f:
            pickle.dump(adj_dict, f)
    
        print(f'Done! Saved {len(adj_dict)} nodes to {out_path}')