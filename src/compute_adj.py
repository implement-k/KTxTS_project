import os, sys, pickle
import geopandas as gpd
import pandas as pd
import json

def compute_adj(year='2023'):
    DATA_DIR = '/Users/implement/KT/KTDB/dataset'
    
    if year == '2019':
        geojson_path = os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20161231.geojson')
    else:
        geojson_path = os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20230101.geojson')
        
    print(f'Loading geojson for {year}...')
    gdf = gpd.read_file(geojson_path)
    
    if year == '2019':
        # Load 10->8 mapping for 2019
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
        
    elif year == '2023':
        manual_dong_mapping = {
            11230740: [11230810],
            31101690: [31101740, 31101750],
            31101700: [31101720, 31101730],
            31103520: [31103620, 31103630],
            31104540: [31104600, 31104610],
            31104590: [31104620, 31104630],
            31250110: [31250600, 31250610, 31250620, 31250630],
        }
        # Reverse map to convert new split codes back to the old base code (to match OD_dong_list_2023)
        reverse_map = {}
        for old_c, new_cs in manual_dong_mapping.items():
            for nc in new_cs:
                reverse_map[nc] = old_c
                
        def convert_code_23(c):
            if pd.isna(c): return 0
            c = int(c) * 10 # adm_cd is 7 digits in the 2023 geojson
            if c in reverse_map: return reverse_map[c]
            return c
            
        gdf['adm_cd'] = pd.to_numeric(gdf['adm_cd'], errors='coerce')
        gdf['adm_cd8'] = gdf['adm_cd'].apply(convert_code_23).astype(int)

    # Merge geometries that map to the same code (dissolve)
    print('Dissolving geometries...')
    gdf = gdf[['adm_cd8', 'geometry']].dissolve(by='adm_cd8').reset_index()

    print('Projecting...')
    gdf = gdf.to_crs(epsg=5179)

    print('Calculating adjacency...')
    sindex = gdf.sindex
    adj_dict = {}

    for i, row in gdf.iterrows():
        code_i = row['adm_cd8']
        geom = row.geometry
        possible_matches_index = list(sindex.intersection(geom.bounds))
        
        adj = []
        for j in possible_matches_index:
            if i == j:
                continue
            other_geom = gdf.iloc[j].geometry
            # Check touches
            if geom.touches(other_geom):
                adj.append(int(gdf.iloc[j]['adm_cd8']))
                
        adj_dict[code_i] = adj

    out_path = os.path.join(DATA_DIR, 'processed', f'dong_adjacency_{year}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(adj_dict, f)

    print(f'Done! Saved {len(adj_dict)} nodes to {out_path}')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        year = sys.argv[1]
    else:
        year = input("년도 선택 (2019/2023):")
    
    if year not in ['2019', '2023']:
        print("잘못된 입력입니다.")
    else:
        compute_adj(year)
