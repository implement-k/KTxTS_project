import pandas as pd
import numpy as np
import os

def process_apartment_ratio(input_path,output_path):
    # 파일 불러오기
    mismatch_df = pd.read_excel("/Users/implement/KT/KTDB/dataset/raw/dong/mismatch_report.xlsx")
    apt_df = pd.read_csv(input_path)
    od_dong_df = pd.read_excel("/Users/implement/KT/KTDB/dataset/raw/OD_dong_list.xlsx")
    valid_od_dongs = set(od_dong_df['dong_code'])
    
    # 신 행정동: 구 행정동 딕셔너리 생성
    mapping_dict = {}
    
    for _, row in mismatch_df.iterrows():
        od_code = row['OD데이터']
        apt_codes_str = str(row['수도권 아파트 비율'])
        
        if pd.isna(od_code) or apt_codes_str == 'nan':
            continue
            
        od_code_int = int(float(od_code))
        
        # 새로 바뀐 행정동 코드의 경우 리스트 추출
        tokens = apt_codes_str.split('/')
        for token in tokens:
            token = token.strip()
            if token.startswith('신'):
                new_code_str = token[1:]
                if new_code_str.isdigit():
                    mapping_dict[int(new_code_str)] = od_code_int
                    
    apt_df['행정구역코드'] = pd.to_numeric(apt_df['행정구역코드'], errors='coerce')
    apt_df['mapped_code'] = apt_df['행정구역코드'].map(lambda x: mapping_dict.get(x, x))
    
    grouped = apt_df.groupby('mapped_code').agg({
        '시도': 'first',
        '시군구명칭': 'first',
        '읍면동명칭': lambda x: '/'.join(x.unique()),
        '아파트수': 'sum',
        '전체주택수': 'sum'
    }).reset_index()
    
    grouped['아파트비율_퍼센트'] = np.where(
        grouped['전체주택수'] > 0,
        (grouped['아파트수'] / grouped['전체주택수']) * 100,
        0
    ).round(2)
    
    grouped.rename(columns={'mapped_code': '행정구역코드'}, inplace=True)
    
    processed_dongs = set(grouped['행정구역코드'])
    
    # 아파트 데이터에는 있지만 OD_dong_list에는 없는 동 (미승인/알 수 없는 동)
    unknown_dongs = processed_dongs - valid_od_dongs
    if unknown_dongs:
        print(f"\nE: OD_dong_list에 존재하지 않는 동이 아파트 데이터에 {len(unknown_dongs)}건 포함")
        unknown_df = grouped[grouped['행정구역코드'].isin(unknown_dongs)]
        for _, row in unknown_df.iterrows():
            print(f"  - 코드: {int(row['행정구역코드'])}, 이름: {row['시도']} {row['시군구명칭']} {row['읍면동명칭']}")
    
    # OD_dong_list에는 있지만 아파트 데이터에는 없는 동 (부족한 동)
    missing_dongs = valid_od_dongs - processed_dongs
    if missing_dongs:
        print(f"\nE: OD_dong_list에 있지만 아파트 데이터에는 누락된 동이 {len(missing_dongs)}건")
        missing_df = od_dong_df[od_dong_df['dong_code'].isin(missing_dongs)]
        for _, row in missing_df.iterrows():
            print(f"  - 누락 코드: {int(row['dong_code'])}, 동이름: {row['dong_name']}")
    
    grouped.drop(columns=['시도', '시군구명칭', '읍면동명칭', '아파트수', '전체주택수'], inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n처리 완료. 기존 {len(apt_df)}개 행 -> {len(grouped)}개 행으로 통합/변환됨.")
    print(f"저장 위치: {output_path}")

if __name__ == "__main__":
    process_apartment_ratio()
