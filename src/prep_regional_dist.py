import os
import pandas as pd
import geopandas as gpd

current_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(current_dir, '..', 'dataset', 'raw')
PROCESSED_DIR = os.path.join(current_dir, '..', 'dataset', 'processed')

def main():
    dong_info_path = os.path.join(RAW_DIR, 'dong_info.geojson')
    
    print("Loading geojson...")
    gdf = gpd.read_file(dong_info_path)
    
    # 한국 좌표계로 투영 변환 (거리 계산을 위해: EPSG:5179)
    if gdf.crs.to_string() != 'EPSG:5179':
        try:
            gdf = gdf.to_crs(epsg=5179)
        except Exception as e:
            print("CRS transform error, fallback to epsg:3857", e)
            gdf = gdf.to_crs(epsg=3857)
    
    gdf['centroid'] = gdf.geometry.centroid
    
    name_to_point = {}
    for idx, row in gdf.iterrows():
        adm_nm = row['adm_nm']
        dong_name = adm_nm.split()[-1]
        name_to_point[dong_name] = row['centroid']
        
    print(f"Total dongs in geojson: {len(name_to_point)}")

    regions = ['jeju', 'busan', 'daegu', 'daejeon', 'gwangju']
    
    for region in regions:
        od_path = os.path.join(RAW_DIR, f'od_{region}.xlsx')
        if not os.path.exists(od_path):
            continue
            
        print(f"--- Processing {region} ---")
        zones = pd.read_excel(od_path, sheet_name='존체계')
        region_dongs = zones['행정동'].tolist()
        
        matched_points = []
        for d in region_dongs:
            if d in name_to_point:
                matched_points.append(name_to_point[d])
            else:
                print(f"  Warning: {d} not found in geojson.")
                matched_points.append(None)
                
        n = len(region_dongs)
        dist_matrix = []
        for i in range(n):
            row_dist = []
            for j in range(n):
                if matched_points[i] is None or matched_points[j] is None:
                    row_dist.append(0.0)
                else:
                    dist = matched_points[i].distance(matched_points[j]) / 1000.0
                    row_dist.append(dist)
            dist_matrix.append(row_dist)
            
        df_dist = pd.DataFrame(dist_matrix, index=zones['권역 존체계_읍면동'], columns=zones['권역 존체계_읍면동'])
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, f'{region}_dist.csv')
        df_dist.to_csv(out_path)
        print(f"Saved {region} distance matrix to {out_path}")

if __name__ == '__main__':
    main()
