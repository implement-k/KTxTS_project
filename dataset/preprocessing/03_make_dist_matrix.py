import os
import pandas as pd
import geopandas as gpd
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # 하버사인
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def make_dist_matrix():
    base_dir = "/Users/implement/KT/KTDB/dataset"
    raw_dir = os.path.join(base_dir, "raw")
    output_path = os.path.join(base_dir, "dist_data.csv")
    
    dong_file = os.path.join(raw_dir, 'OD_dong_list.xlsx')
    dong_df = pd.read_excel(dong_file)
    valid_dongs = dong_df['dong_code'].astype(int).values
    
    # Load Change Review 
    change_review_path = os.path.join(raw_dir, 'dong', 'hangjeongdong_change_review.csv')
    od_dong_change_list = pd.read_csv(change_review_path)
    change_map = dict(zip(od_dong_change_list['이후_행정동코드'], od_dong_change_list['이전_행정동코드']))
    
    # Load GeoJSON 
    geojson_path = os.path.join(raw_dir, 'dong', 'dong_area_20220101.geojson')
    gdf = gpd.read_file(geojson_path)
    
    # 7자리 -> 8자리
    gdf['adm_cd_8digit'] = pd.to_numeric(gdf['adm_cd'], errors='coerce') * 10
    
    # 행정동 중심좌표 계산
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf['centroid'] = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    gdf['lon'] = gdf['centroid'].x
    gdf['lat'] = gdf['centroid'].y
    
    # 행정동 코드와 중심좌표를 매핑
    coords_dict = dict(zip(gdf['adm_cd_8digit'], zip(gdf['lat'], gdf['lon'])))
    
    num_dongs = len(valid_dongs)
    coords = np.zeros((num_dongs, 2), dtype=np.float32)
    for i, code in enumerate(valid_dongs):
        coords[i, 0] = coords_dict[code][0]
        coords[i, 1] = coords_dict[code][1]
        
    # 행정동간 거리 계산
    lat1 = coords[:, 0][:, np.newaxis]
    lon1 = coords[:, 1][:, np.newaxis]
    lat2 = coords[:, 0][np.newaxis, :]
    lon2 = coords[:, 1][np.newaxis, :]
    
    X_distance = haversine(lon1, lat1, lon2, lat2)
    
    # 내부 거리는 1km로 설정
    for i in range(num_dongs):
        X_distance[i, i] = 1
        
    # Meshgrid를 사용하여 O, D 조합 생성
    O_grid, D_grid = np.meshgrid(valid_dongs, valid_dongs, indexing='ij')
    
    # 1D 배열로 평탄화
    df_dist = pd.DataFrame({
        'O_dong_code': O_grid.flatten(),
        'D_dong_code': D_grid.flatten(),
        'distance': X_distance.flatten()
    })
    
    # 소수점 3자리로 반올림
    df_dist['distance'] = df_dist['distance'].round(3)
    
    df_dist.to_csv(output_path, index=False)
    print(f"\n완료! 총 {len(df_dist):,}개의 O-D 거리 쌍이 성공적으로 저장되었습니다.")
    print(f"저장 위치: {output_path}")

if __name__ == '__main__':
    make_dist_matrix()
