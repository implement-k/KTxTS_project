import os
import pandas as pd
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
folders = ['all', 'changneung', 'wangsuk', 'gyosan']

manual_dong_mapping = {
    11230740: [11230810],
    31101690: [31101740, 31101750],
    31101700: [31101720, 31101730],
    31103520: [31103620, 31103630],
    31104540: [31104600, 31104610],
    31104590: [31104620, 31104630],
    31250110: [31250600, 31250610, 31250620, 31250630],
}

def normalize_codes(codes_set):
    normalized = set()
    for c in codes_set:
        try:
            c = int(float(c))
        except (ValueError, TypeError):
            continue
            
        found = False
        # 매핑 사전에 있으면 대표 코드(key)로 통일
        for key, values in manual_dong_mapping.items():
            if c in values:
                normalized.add(key)
                found = True
                break
        
        # 값이 없었거나, key 본인이면 원래 값 그대로 넣음
        if not found:
            normalized.add(c)
            
    return normalized

for folder in folders:
    print(f"========== Checking folder: {folder} ==========")
    excel_path = os.path.join(base_dir, folder, 'OD_dong_list_2023.xlsx')
    geo_path = os.path.join(base_dir, folder, 'dong_area.geojson')
    
    if not os.path.exists(excel_path):
        print(f"E: {excel_path} 파일이 없습니다.")
        continue
    if not os.path.exists(geo_path):
        print(f"E: {geo_path} 파일이 없습니다.")
        continue
    
    # 파일 열기
    dong_df = pd.read_excel(excel_path)
    with open(geo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    codes = set()
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        code_str = str(props.get('ADM_CD', '')).strip()
        
        # 7자리 코드인 경우 뒤에 '0' 추가
        if len(code_str) == 7 and code_str.isdigit():
            code_str += '0'
        codes.add(code_str)
    
    dong_raw = set(dong_df['dong_code'].dropna().unique())
    geo_raw = codes
    
    # print(len(dong_raw), "개 동 코드 (OD_dong_list_2023.xlsx)")
    # print(len(geo_raw), "개 동 코드 (dong_area.geojson)")
    
    dong_norm = normalize_codes(dong_raw)
    geo_norm = normalize_codes(geo_raw)
    
    only_in_dong = dong_norm - geo_norm
    only_in_geo = geo_norm - dong_norm
    
    if len(only_in_dong) == 0 and len(only_in_geo) == 0:
        print("✅ MATCH! (매핑 후 OD_dong_list와 dong_area의 동 코드가 완벽히 일치합니다.)")
    else:
        print("❌ MISMATCH DETECTED (geojson)!")
        if only_in_dong:
            print(f" - OD_dong_list 에만 있는 코드 ({len(only_in_dong)}개): {sorted(list(only_in_dong))}")
        if only_in_geo:
            print(f" - dong_area.geojson 에만 있는 코드 ({len(only_in_geo)}개): {sorted(list(only_in_geo))}")
            
    # Check static features
    for static_file in ['static_features_initial.csv', 'static_features_middle.csv', 'static_features_final.csv']:
        static_path = os.path.join(base_dir, folder, static_file)
        if not os.path.exists(static_path):
            continue
            
        static_df = pd.read_csv(static_path, encoding='utf-8-sig')
        static_raw = set(static_df['dong_code'].dropna().unique())
        static_norm = normalize_codes(static_raw)
        
        only_in_dong_vs_static = dong_norm - static_norm
        only_in_static = static_norm - dong_norm
        
        if len(only_in_dong_vs_static) == 0 and len(only_in_static) == 0:
            print(f"✅ MATCH! (매핑 후 OD_dong_list와 {static_file}의 동 코드가 완벽히 일치합니다.)")
        else:
            print(f"❌ MISMATCH DETECTED ({static_file})!")
            if only_in_dong_vs_static:
                print(f" - OD_dong_list 에만 있는 코드 ({len(only_in_dong_vs_static)}개): {sorted(list(only_in_dong_vs_static))}")
            if only_in_static:
                print(f" - {static_file} 에만 있는 코드 ({len(only_in_static)}개): {sorted(list(only_in_static))}")
            
    print()