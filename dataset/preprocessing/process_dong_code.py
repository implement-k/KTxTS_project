import pandas as pd

def check_dong(df, dong_cal_name):
    od_dong_list = pd.read_excel("/Users/implement/KT/KTDB/dataset/raw/OD_dong_list.xlsx")
    
    # 7자리 코드 8자리 코드로 변환
    df_code = pd.to_numeric(df[dong_cal_name], errors='coerce')
    df_code_8digit = df_code.mask(df_code < 10000000, df_code * 10)
    
    valid_dongs = set(od_dong_list['dong_code'])
    
    invalid_mask = df_code_8digit.isna() | ~df_code_8digit.isin(valid_dongs)
    
    invalid_rows = df[invalid_mask]
    if not invalid_rows.empty:
        invalid_unique_codes = invalid_rows[dong_cal_name].unique()
        print(f"E: 행정동코드 누락 또는 존재하지 않음. 총 {len(invalid_rows)}건 skip 처리됨.")
        print(f"   제외된 코드 목록(원본): {invalid_unique_codes}")
        
    df_filtered = df[~invalid_mask].copy()
    
    # 8자리 표준 코드로 변환된 값을 최종 결과에 반영
    df_filtered[dong_cal_name] = df_code_8digit[~invalid_mask].astype(int)
    
    return df_filtered
