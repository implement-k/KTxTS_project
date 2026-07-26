import pandas as pd
import os
def make_od_matrix(year='2023'):
    if year == '2023':
        input_file = "/Users/implement/KT/KTDB/dataset/raw/ODTRIP23_F.OUT"
        od_csv_output = "/Users/implement/KT/KTDB/dataset/od_data_2023.csv"
        od_static_output = "/Users/implement/KT/KTDB/dataset/od_static_feature_2023.csv"
    elif year == '2019':
        input_file = "/Users/implement/KT/KTDB/dataset/raw/ODTRIP19_F.OUT"
        od_csv_output = "/Users/implement/KT/KTDB/dataset/od_data_2019.csv"
        od_static_output = "/Users/implement/KT/KTDB/dataset/od_static_feature_2019.csv"
    else:
        raise ValueError("Unsupported year. Please choose '2019' or '2023'.")

    print(f"파일 읽기 시작: {input_file}")
    
    if year == '2019':
        # 2019 has 14 columns: O_index, O_dong_code, O_zone, D_index, D_dong_code, D_zone, and 8 purposes
        columns_19 = [
            'O_index', 'O_dong_code', 'O_zone',
            'D_index', 'D_dong_code', 'D_zone',
            '귀가', '출근', '등교', '학원', '업무', '쇼핑', '여가', '기타'
        ]
        df = pd.read_csv(input_file, sep=r',?\s+', names=columns_19, engine='python')
        
        # '학원', '쇼핑', '여가'를 '기타'에 합침
        df['기타'] = df['기타'] + df['학원'] + df['쇼핑'] + df['여가']
        
        # 불필요한 컬럼 삭제하여 2023 형식과 맞춤
        df = df[['O_index', 'O_dong_code', 'D_index', 'D_dong_code', '귀가', '출근', '등교', '업무', '기타']]
    else:
        columns_23 = [
            'O_index', 'O_dong_code', 
            'D_index', 'D_dong_code', 
            '귀가', '출근', '등교', '업무', '기타'
        ]
        df = pd.read_csv(input_file, sep=r'\s+', names=columns_23, engine='c')
    
    if year == '2019':
        # 1. 10자리 -> 8자리 완벽 사전 생성 및 로드
        mapping_path = "/Users/implement/KT/KTDB/dataset/preprocessing/process/mapping_2019_10_to_8.json"
        import json
        with open(mapping_path, 'r', encoding='utf-8') as f:
            c10_to_c8_str = json.load(f)
        c10_to_c8 = {int(k): int(v) for k, v in c10_to_c8_str.items()}
        
        # 10자리 비수도권 코드(매핑되지 않은 코드)를 8자리 시도 체계로 임시 변환
        sido_10_to_8 = {
            26: 21, 27: 22, 29: 24, 30: 25, 31: 26, 36: 29,
            42: 32, 43: 33, 44: 34, 45: 35, 46: 36, 47: 37, 48: 38, 50: 39
        }

        def convert_code(c):
            if pd.isna(c): return 0
            c = int(c)
            # 1. 사전 매핑 확인
            if c in c10_to_c8:
                return c10_to_c8[c]
                        
            # 2. 매핑에 없는 10자리 코드 중 비수도권인 경우
            if c >= 1000000000:
                sido_10 = int(c // 100000000)
                if sido_10 in sido_10_to_8:
                    return sido_10_to_8[sido_10] * 1000000
            
            # 3. 전부 실패한 경우 그대로 반환 (혹은 이미 8자리인 경우 그대로)
            return c
            
        o_code = pd.to_numeric(df['O_dong_code'], errors='coerce')
        d_code = pd.to_numeric(df['D_dong_code'], errors='coerce')
        
        df['O_dong_code'] = o_code.apply(convert_code).astype(int)
        df['D_dong_code'] = d_code.apply(convert_code).astype(int)

    else:
        # 동 코드 7자리 -> 8자리 변환
        o_code = pd.to_numeric(df['O_dong_code'], errors='coerce')
        df['O_dong_code'] = o_code.mask(o_code < 10000000, o_code * 10).fillna(0).astype(int)
        
        d_code = pd.to_numeric(df['D_dong_code'], errors='coerce')
        df['D_dong_code'] = d_code.mask(d_code < 10000000, d_code * 10).fillna(0).astype(int)
    
    # 총 통행량 계산 (5개 목적 합산)
    df['total_trips'] = df[['귀가', '출근', '등교', '업무', '기타']].sum(axis=1)
    
    # 시도 코드 -> 이름 매핑 딕셔너리
    sido_map = {
        21: '부산', 22: '대구', 24: '광주', 25: '대전', 26: '울산', 29: '세종',
        32: '강원', 33: '충북', 34: '충남', 35: '전북', 36: '전남', 37: '경북', 38: '경남', 39: '제주'
    }
    
    # 1. 수도권(O) -> 비수도권(D) 통행
    cap_to_ext = df[['O_dong_code', 'D_dong_code', 'total_trips']].copy()
    cap_to_ext['D_sido_name'] = (cap_to_ext['D_dong_code'] // 1000000).map(sido_map)
    d_features = cap_to_ext.dropna(subset=['D_sido_name']).groupby(['O_dong_code', 'D_sido_name'])['total_trips'].sum().unstack(fill_value=0)
    d_features.columns = [f'd_{col}' for col in d_features.columns]
    
    # 2. 비수도권(O) -> 수도권(D) 통행
    ext_to_cap = df[['O_dong_code', 'D_dong_code', 'total_trips']].copy()
    ext_to_cap['O_sido_name'] = (ext_to_cap['O_dong_code'] // 1000000).map(sido_map)
    o_features = ext_to_cap.dropna(subset=['O_sido_name']).groupby(['D_dong_code', 'O_sido_name'])['total_trips'].sum().unstack(fill_value=0)
    o_features.columns = [f'o_{col}' for col in o_features.columns]
    
    # 3. 수도권 동 기준으로 병합
    all_dongs = set(df['O_dong_code'].unique()) | set(df['D_dong_code'].unique())
    capital_dongs = [d for d in all_dongs if (d // 1000000) in [11, 23, 31]]
    static_df = pd.DataFrame({'dong_code': capital_dongs})
    static_df = static_df.merge(o_features, left_on='dong_code', right_index=True, how='left')
    static_df = static_df.merge(d_features, left_on='dong_code', right_index=True, how='left')
    static_df.fillna(0, inplace=True)
    
    # 정수형 변환
    for col in static_df.columns:
        if col != 'dong_code':
            static_df[col] = static_df[col].astype(int)
            
    os.makedirs(os.path.dirname(od_static_output), exist_ok=True)
    static_df.to_csv(od_static_output, index=False, encoding='utf-8-sig')
    print(f"Static Feature 생성 완료: {od_static_output}")
    
    # 기존 처리: 수도권 내부(11, 23, 31) 통행만 남기고 필터링
    is_o_capital = (df['O_dong_code'] // 1000000).isin([11, 23, 31])
    is_d_capital = (df['D_dong_code'] // 1000000).isin([11, 23, 31])
    df_filtered = df[is_o_capital & is_d_capital].copy()
    
    # 불필요한 임시 컬럼 제거
    df_filtered.drop(columns=['total_trips'], inplace=True)
    
    for col in ['귀가', '출근', '등교', '업무', '기타']:
        df_filtered[col] = df_filtered[col].astype('float32')
    
    for col in ['O_index', 'D_index']:
        df_filtered[col] = df_filtered[col].astype('int32')
        
    df_filtered.to_csv(od_csv_output, index=False)
    
    print(f"정제된 OD 매트릭스 저장 완료: {od_csv_output}")
    filtered_dongs = set(df_filtered['O_dong_code'].unique()) | set(df_filtered['D_dong_code'].unique())
    print(f"\n=== 처리 결과 요약 ({year}년) ===")
    print(f"1. 전체 동(수도권+비수도권) 개수: {len(all_dongs)}")
    print(f"2. 수도권 동(static feature 대상) 개수: {len(capital_dongs)}")
    print(f"3. 수도권 내부 OD 필터링 후 실제 등장한 동 개수: {len(filtered_dongs)}")
    if len(capital_dongs) != len(filtered_dongs):
        diff = set(capital_dongs) - filtered_dongs
        print(f"   ⚠ static_df엔 있지만 필터링된 OD엔 안 나온 동: {len(diff)}개")

if __name__ == "__main__":
    year = input("년도 선택 (2019/2023):")
    if year not in ['2019', '2023']:
        print("잘못된 입력입니다. '2019' 또는 '2023'을 입력해주세요.")
    else:
        make_od_matrix(year)
