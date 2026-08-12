import geopandas as gpd
import json
import os
import pandas as pd

print('Loading geojson...')
gdf = gpd.read_file('/Users/implement/KT/KTDB/dataset/raw/dong/dong_area_20161231.geojson')

# Load mapping
mapping_path = "/Users/implement/KT/KTDB/dataset/preprocessing/process/mapping_2019_10_to_8.json"
with open(mapping_path, 'r', encoding='utf-8') as f:
    c10_to_c8_str = json.load(f)
c10_to_c8 = {int(k): int(v) for k, v in c10_to_c8_str.items()}

# Function to map code
def convert_code(c):
    if pd.isna(c): return 0
    c = int(c)
    if c in c10_to_c8:
        return c10_to_c8[c]
    return c

gdf['adm_cd'] = pd.to_numeric(gdf['adm_cd'], errors='coerce')
gdf['adm_cd8'] = gdf['adm_cd'].apply(convert_code).astype(int)

# Group by mapped code and dissolve geometries
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
        if i == j: continue
        other_geom = gdf.iloc[j].geometry
        if geom.touches(other_geom):
            adj.append(int(gdf.iloc[j]['adm_cd8']))
    adj_dict[code_i] = adj

out_path = '/Users/implement/KT/KTDB/dataset/processed/dong_adjacency_2019.pkl'
import pickle
with open(out_path, 'wb') as f:
    pickle.dump(adj_dict, f)
print(f'Done! Saved {len(adj_dict)} nodes to {out_path}')
