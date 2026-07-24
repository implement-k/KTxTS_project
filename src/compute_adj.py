import os, sys, pickle
import geopandas as gpd

DATA_DIR = '/Users/implement/KT/KTDB/dataset'
geojson_path = os.path.join(DATA_DIR, 'raw', 'dong', 'dong_area_20220101.geojson')

print('Loading geojson...')
gdf = gpd.read_file(geojson_path)
gdf['adm_cd8'] = gdf['adm_cd8'].astype(int)

# It's highly recommended to project to a metric CRS before checking touches/bounds, but EPSG 5179 is good.
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

out_path = os.path.join(DATA_DIR, 'processed', 'dong_adjacency.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(adj_dict, f)

print(f'Done! Saved {len(adj_dict)} nodes to {out_path}')
