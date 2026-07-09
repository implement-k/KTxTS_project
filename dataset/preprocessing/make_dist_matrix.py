import os
import pandas as pd
import geopandas as gpd
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '../..')
    ktdb_dir = os.path.join(base_dir, 'KTDB')
    
    # 1. Load dong_code.xlsx to get the 1153 7-digit codes and define indices
    dong_file = os.path.join(base_dir, 'dataset', 'dong_code.xlsx')
    dong_df = pd.read_excel(dong_file)
    dong_codes_7 = dong_df['읍면동'].astype(str).values
    idx_map = {code: i for i, code in enumerate(dong_codes_7)}
    
    # 2. Load GeoJSON to get centroids
    print("Loading GeoJSON...")
    geojson_path = os.path.join(base_dir, 'dataset', 'HangJeongDong_ver20230101.geojson')
    gdf = gpd.read_file(geojson_path)
    gdf['adm_cd'] = gdf['adm_cd'].astype(str)
    
    # Calculate Centroids (Using projected CRS for accuracy)
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf['centroid'] = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    gdf['lon'] = gdf['centroid'].x
    gdf['lat'] = gdf['centroid'].y
    
    # 3. Create mapping from 7-digit code to lat/lon
    coord_map = dict(zip(gdf['adm_cd'], zip(gdf['lat'], gdf['lon'])))
    
    # 4. Fill coordinate array for 1153 zones
    num_dongs = len(dong_codes_7)
    coords = np.zeros((num_dongs, 2), dtype=np.float32)
    
    for i, code in enumerate(dong_codes_7):
        if code in coord_map:
            coords[i, 0] = coord_map[code][0] # lat
            coords[i, 1] = coord_map[code][1] # lon
        else:
            # Fallback to mean of that Sigungu if Dong is missing
            sigungu = code[:5]
            match = gdf[gdf['adm_cd'].str.startswith(sigungu)]
            if not match.empty:
                coords[i, 0] = match['lat'].mean()
                coords[i, 1] = match['lon'].mean()
            else:
                # If utterly failed, default to Seoul center
                coords[i, 0] = 37.5665
                coords[i, 1] = 126.9780
                
    # 5. Compute pairwise Haversine Distance Matrix
    print("Computing Distance Matrix...")
    X_distance = np.zeros((num_dongs, num_dongs), dtype=np.float32)
    
    # Broadcasting to calculate distances quickly
    lat1 = coords[:, 0][:, np.newaxis]
    lon1 = coords[:, 1][:, np.newaxis]
    lat2 = coords[:, 0][np.newaxis, :]
    lon2 = coords[:, 1][np.newaxis, :]
    
    X_distance = haversine(lon1, lat1, lon2, lat2)
    
    # Intrazonal distance (approximate based on area or just set to 1.5km as standard)
    for i in range(num_dongs):
        X_distance[i, i] = 1.5 # 1.5km for intra-zonal trip roughly
        
    out_path = os.path.join(ktdb_dir, 'dataset', 'X_distance.npy')
    np.save(out_path, X_distance)
    print(f"✅ Saved X_distance to {out_path} with shape {X_distance.shape}")

if __name__ == '__main__':
    main()
