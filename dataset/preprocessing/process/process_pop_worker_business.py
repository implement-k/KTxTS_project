import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_dong_code as pdc

def process_pop_worker_business(input_path, output_path, year='2023'):
    print(f"processing population worker business data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)
    
    df = df.drop(columns=['population_year', 'business_year', 'sigungu', 'sido', 'dong_name'], errors='ignore')
    
    # map_codes_from_mismatch를 통해 1:N 매핑 처리
    mismatch_col = 'Population_Business_Worker_2021_2019' if str(year) == '2019' else '행정동별 연령대별 인구, 종사자수, 사업체수 2024년 데이터'
    df_mapped, valid_dongs = pdc.map_codes_from_mismatch(df, 'dong_code', mismatch_col, year=str(year))
    
    if not df_mapped.empty:
        # dong_code 기준으로 합산(sum)하여 중복(복제)된 인구/사업체 데이터를 나눔
        # 복제된 경우, 면적 비율로 나누는 대신 단순히 N등분 (평균)하거나, 혹은 원본값 그대로 유지할지 결정
        # 이 데이터셋은 개수를 의미하므로, 1:N으로 복제된 경우 원래 개수를 N등분해야 맞음.
        # 하지만 기존 로직에서는 단순 sum이나 groupby sum을 했기 때문에 문제가 발생.
        # 따라서 N등분 처리가 필요함!
        
        # 각 raw_code별 분할 개수 구하기 위해 잠시 복구
        # (단 map_codes_from_mismatch에서 중복을 나눠주는 기능은 없으므로 여기서 나눔)
        df_mapped['mapped_code'] = df_mapped['mapped_code'].astype(int)
        
        # count how many times each original row was duplicated (by index)
        duplicate_counts = df_mapped.groupby(df_mapped.index).size()
        for count_col in ['pop_0_19', 'pop_20_59', 'pop_60_plus', 'worker_count', 'business_count']:
            if count_col in df_mapped.columns:
                df_mapped[count_col] = df_mapped[count_col] / df_mapped.index.map(duplicate_counts)
        
        cols_to_keep = ['mapped_code', 'pop_0_19', 'pop_20_59', 'pop_60_plus', 'worker_count', 'business_count']
        cols_to_keep = [c for c in cols_to_keep if c in df_mapped.columns]
        grouped = df_mapped[cols_to_keep].groupby('mapped_code').sum().reset_index()
        grouped.rename(columns={'mapped_code': 'dong_code'}, inplace=True)
        
        # Ensure only valid dongs
        invalid_mask = ~grouped['dong_code'].isin(valid_dongs)
        if invalid_mask.any():
            print(f"E: 유효하지 않은 코드 병합됨 ({invalid_mask.sum()}건)")
            
        grouped = grouped[~invalid_mask].copy()
        
        # Int 변환
        for col in grouped.columns:
            grouped[col] = grouped[col].round().astype(int)
            
        df = grouped
    else:
        # Fallback if map_codes_from_mismatch returns empty
        df = pdc.check_dong(df, 'dong_code', year)

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 결과 저장: {output_path}, 동 개수 : {len(df)}")
    print("모든 처리 완료.")

if __name__ == "__main__":
    input_file_2023 = "/Users/implement/KT/KTDB/dataset/raw/2021-2023 인구 및 사업자 데이터.csv"
    output_file_2023 = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2023.csv"
    process_pop_worker_business(input_file_2023, output_file_2023, year='2023')
    
    input_file_2019 = "/Users/implement/KT/KTDB/dataset/raw/Population Business Worker 2021 2019.csv"
    output_file_2019 = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2019.csv"
    process_pop_worker_business(input_file_2019, output_file_2019, year='2019')
