import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_dong_code as pdc

def process_apartment_ratio(input_path, output_path, year='2023'):
    print(f"processing apartment ratio data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)
    
    # map_codes_from_mismatch를 통해 1:N 매핑 처리
    mismatch_col = '수도권_아파트_비율_2019' if str(year) == '2019' else '수도권 아파트 비율'
    df_mapped, valid_dongs = pdc.map_codes_from_mismatch(df, '행정구역코드', mismatch_col, year=str(year))
    
    if not df_mapped.empty:
        df_mapped['mapped_code'] = df_mapped['mapped_code'].astype(int)
        
        # 아파트 비율은 %이므로 1:N으로 복제되더라도 그대로 유지 (평균값 사용)
        cols_to_keep = ['mapped_code', '아파트비율_퍼센트']
        cols_to_keep = [c for c in cols_to_keep if c in df_mapped.columns]
        grouped = df_mapped[cols_to_keep].groupby('mapped_code').mean().reset_index()
        grouped.rename(columns={'mapped_code': '행정구역코드'}, inplace=True)
        
        invalid_mask = ~grouped['행정구역코드'].isin(valid_dongs)
        if invalid_mask.any():
            print(f"E: 유효하지 않은 코드 병합됨 ({invalid_mask.sum()}건)")
            
        grouped = grouped[~invalid_mask].copy()
        
        # Omission (누락된 동) 0으로 채우기
        missing_dongs = valid_dongs - set(grouped['행정구역코드'])
        if missing_dongs:
            print(f"I: OD_dong_list에 있지만 아파트 데이터에 누락된 동이 {len(missing_dongs)}건. 0으로 패딩 처리합니다.")
            missing_df = pd.DataFrame([{'행정구역코드': c, '아파트비율_퍼센트': 0.0} for c in missing_dongs])
            grouped = pd.concat([grouped, missing_df], ignore_index=True)
            
        # Int 변환 (비율은 float 유지)
        grouped['행정구역코드'] = grouped['행정구역코드'].astype(int)
            
        df = grouped
    else:
        # Fallback
        df = pdc.check_dong(df, '행정구역코드', year)

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 결과 저장: {output_path}, 동 개수: {len(df)}")
    print("모든 처리 완료.")

if __name__ == "__main__":
    input_file_2023 = "/Users/implement/KT/KTDB/dataset/raw/서울 인천 경기 아파트 비율 2024.csv"
    output_file_2023 = "/Users/implement/KT/KTDB/dataset/processed/processed_apartment_ratio_2023.csv"
    process_apartment_ratio(input_file_2023, output_file_2023, year='2023')
    
    input_file_2019 = "/Users/implement/KT/KTDB/dataset/raw/서울 인천 경기 아파트 비율 2019.csv"
    output_file_2019 = "/Users/implement/KT/KTDB/dataset/processed/processed_apartment_ratio_2019.csv"
    process_apartment_ratio(input_file_2019, output_file_2019, year='2019')
