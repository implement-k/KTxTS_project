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

def make_dist_matrix(dong_path, geo_path):
    dong_df = pd.read_excel(dong_path)
    valid_dongs = dong_df['dong_code'].astype(int).values
    
    # Load GeoJSON 
    gdf = gpd.read_file(geo_path)
    
    # geojson의 코드가 7자리인 경우에만 끝에 '0'을 붙여 8자리로 보정
    gdf['ADM_CD'] = gdf['ADM_CD'].apply(lambda x: str(x).strip() + '0' if len(str(x).strip()) == 7 else str(x))
    gdf['ADM_CD'] = pd.to_numeric(gdf['ADM_CD'], errors='coerce')
    
    # 행정동 중심좌표 계산
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf['centroid'] = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    gdf['lon'] = gdf['centroid'].x
    gdf['lat'] = gdf['centroid'].y
    
    # 행정동 코드와 중심좌표를 매핑
    coords_dict = dict(zip(gdf['ADM_CD'], zip(gdf['lat'], gdf['lon'])))
    
    manual_dong_mapping = {
        11230740: [11230810],              # 일원2동 -> 개포3동
        31101690: [31101740, 31101750],    # 행신3동 -> 행신3동, 행신4동
        31101700: [31101720, 31101730],    # 삼송동 -> 삼송1동, 삼송2동
        31103520: [31103620, 31103630],    # 중산동 -> 중산1동, 중산2동
        31104540: [31104600, 31104610],    # 탄현동 -> 탄현1동, 탄현2동
        31104590: [31104620, 31104630],    # 송산동 -> 덕이동, 가좌동
        31250110: [31250600, 31250610, 31250620, 31250630],  # 오포읍 -> 오포1동, 오포2동, 신현동, 능평동
    }

    manual_mapped_coords = {}
    for old_code, sub_codes in manual_dong_mapping.items():
        sub_gdf = gdf_proj[gdf_proj['ADM_CD'].isin(sub_codes)]
        
        missing = [c for c in sub_codes if c not in sub_gdf['ADM_CD'].values]
        if missing:
            print(f"  ⚠ 수동 매핑 경고: {old_code}의 하위 동코드 {missing}가 GeoJSON에 없어 일부만 반영됨")
            
        if not sub_gdf.empty:
            merged_geom = sub_gdf.geometry.unary_union
            centroid = merged_geom.centroid
            centroid_4326 = gpd.GeoSeries([centroid], crs=gdf_proj.crs).to_crs(epsg=4326).iloc[0]
            manual_mapped_coords[old_code] = (centroid_4326.y, centroid_4326.x)
            
    # 시군구 평균 좌표 계산 (Fallback 용)
    sigungu_coords = {}
    for code, (lat, lon) in coords_dict.items():
        if pd.isna(code): continue
        sigungu = int(code // 1000)
        if sigungu not in sigungu_coords:
            sigungu_coords[sigungu] = []
        sigungu_coords[sigungu].append((lat, lon))
        
    sigungu_mean_coords = {
        s: (np.mean([x[0] for x in lst]), np.mean([x[1] for x in lst]))
        for s, lst in sigungu_coords.items()
    }
    overall_mean = (gdf['lat'].mean(), gdf['lon'].mean())
    
    num_dongs = len(valid_dongs)
    coords = np.zeros((num_dongs, 2), dtype=np.float32)
    for i, code in enumerate(valid_dongs):
        if code in coords_dict:
            coords[i, 0] = coords_dict[code][0]
            coords[i, 1] = coords_dict[code][1]
        elif code in manual_mapped_coords:
            coords[i, 0] = manual_mapped_coords[code][0]
            coords[i, 1] = manual_mapped_coords[code][1]
        else:
            sigungu = int(code // 1000)
            if sigungu in sigungu_mean_coords:
                coords[i, 0] = sigungu_mean_coords[sigungu][0]
                coords[i, 1] = sigungu_mean_coords[sigungu][1]
            else:
                coords[i, 0] = overall_mean[0]
                coords[i, 1] = overall_mean[1]
        
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
    O_grid, D_grid = np.meshgrid(valid_dongs, valid_dongs, indexing='ij') # type: ignore
    
    # 1D 배열로 평탄화
    df_dist = pd.DataFrame({
        'O_dong_code': O_grid.flatten(),
        'D_dong_code': D_grid.flatten(),
        'distance': X_distance.flatten()
    })
    
    # 소수점 3자리로 반올림
    df_dist['distance'] = df_dist['distance'].round(3)
    # dong_code -> dong_name 매핑 (fallback 리스트 출력용)
    dong_code_to_name = dict(zip(dong_df['dong_code'].astype(int), dong_df['dong_name']))

    matched_count = sum(1 for code in valid_dongs if code in coords_dict)
    manual_matched_count = sum(1 for code in valid_dongs if code not in coords_dict and code in manual_mapped_coords)
    sigungu_fallback_count = sum(
        1 for code in valid_dongs
        if code not in coords_dict and code not in manual_mapped_coords and int(code // 1000) in sigungu_mean_coords
    )
    overall_fallback_count = num_dongs - matched_count - manual_matched_count - sigungu_fallback_count

    print(f"\n=== 처리된 동 개수 요약 ===")
    print(f"1. dong_code 목록 기준 전체 동 개수: {num_dongs}")
    print(f"2. GeoJSON에서 좌표 직접 매칭된 동 개수: {matched_count} ({matched_count/num_dongs*100:.1f}%)")
    print(f"3. 수동 매핑(분동/개명)으로 해결된 동 개수: {manual_matched_count}")
    print(f"4. 시군구 평균 좌표로 fallback된 동 개수: {sigungu_fallback_count}")
    print(f"5. 전체 평균 좌표로 fallback된 동 개수: {overall_fallback_count}")

    # === fallback된 동 (dong_code 목록엔 있는데 GeoJSON에도, 수동매핑에도 없는 동) ===
    fallback_codes = [code for code in valid_dongs if code not in coords_dict and code not in manual_mapped_coords]
    if fallback_codes:
        print(f"\n⚠ Fallback된 동 목록 ({len(fallback_codes)}개):")
        for code in fallback_codes:
            name = dong_code_to_name.get(code, "(이름 없음)")
            sigungu = int(code // 1000)
            method = "시군구 평균" if sigungu in sigungu_mean_coords else "전체 평균"
            print(f"  - {code} ({name}) -> {method}")
    else:
        print("\n✅ Fallback된 동 없음 (전부 GeoJSON 직접매칭 또는 수동매핑으로 해결됨)")
    return df_dist

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = ['all', 'changneung', 'wangsuk', 'gyosan']
    
    for folder in folders:
        print(f"========== Checking folder: {folder} ==========")
        dong_path = os.path.join(base_dir, folder, 'OD_dong_list_2023.xlsx')
        geo_path = os.path.join(base_dir, folder, 'dong_area.geojson')

        if not os.path.exists(geo_path):
            print(f"E: {geo_path} 파일이 없습니다.")
            continue
            
        df_dist= make_dist_matrix(dong_path, geo_path)
        out_path = os.path.join(base_dir, folder, f'dong_distance.csv')
        
        df_dist.to_csv(out_path, index=False)
        print(f"\n완료! 총 {len(df_dist):,}개의 O-D 거리 쌍이 성공적으로 저장되었습니다.")
        print(f"저장 위치: {out_path}")
    
        
