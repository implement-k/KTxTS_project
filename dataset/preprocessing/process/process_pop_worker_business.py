import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import dataset.preprocessing.process.process_dong_code as pdc

def process_pop_worker_business(input_path, output_path, year='2023'):
    print(f"processing population worker business data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)        
    df = df.drop(columns=['population_year', 'business_year', 'sigungu', 'sido', 'dong_name'], errors='ignore')
    # 7자리 코드 8자리 코드로 변환 (process_dong_code.py와 동일 로직)
    df_code = pd.to_numeric(df['dong_code'], errors='coerce')
    df['dong_code'] = df_code.mask(df_code < 10000000, df_code * 10).fillna(0).astype(int).astype(str)
    
    # 2019년 데이터의 경우 매핑 처리된 데이터가 있으므로 검증할 필요 없음.
    if str(year) == '2019':
        base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        od_dong_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
        mismatch_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"mismatch_report_{year}.xlsx"))
        valid_od_dong_codes = set(od_dong_df['dong_code'].astype(str))
        
        def clean_code(x):
            if pd.isna(x): return 'nan'
            try: return str(int(float(x)))
            except: return str(x).strip()
            
        # dong_code -> representative code 매핑
        code_10_to_rep_code = {clean_code(k): clean_code(v) for k, v in zip(od_dong_df['dong_code_10'], od_dong_df['dong_code']) if clean_code(k) != 'nan'}
        code_8_to_rep_code = {clean_code(k): clean_code(v) for k, v in zip(od_dong_df['dong_code'], od_dong_df['dong_code'])}
        code_to_rep_code = {**code_10_to_rep_code, **code_8_to_rep_code}
        data_code_to_rep_code = {}
        
        for _, row in mismatch_df.iterrows():
            od_code_val = row['OD데이터']
            codes_str = str(row['Population_Business_Worker_2021_2019'])
            
            if pd.isna(od_code_val) or codes_str == 'nan':
                continue
                
            od_code_str = clean_code(od_code_val)
            if od_code_str not in code_to_rep_code: continue
            rep_code = code_to_rep_code[od_code_str]
            
            tokens = codes_str.split('/')
            for token in tokens:
                token = token.strip()
                new_code = token.replace('신', '') if token.startswith('신') else token
                if new_code.isdigit():
                    data_code_to_rep_code[new_code] = rep_code
        
        df['mapped_code'] = df['dong_code'].map(lambda x: data_code_to_rep_code.get(clean_code(x), clean_code(x)))
        
        # 합산 처리
        grouped = df.groupby('mapped_code').agg({
            'pop_0_19': 'sum',
            'pop_20_59': 'sum',
            'pop_60_plus': 'sum',
            'worker_count': 'sum',
            'business_count': 'sum'
        }).reset_index()
        
        grouped.rename(columns={'mapped_code': 'dong_code'}, inplace=True)
        
        processed_dongs:set[str] = set(grouped['dong_code'])
        
        unknown_dongs = processed_dongs - valid_od_dong_codes
        if unknown_dongs:
            print(f"\nE: OD_dong_list에 존재하지 않는 동이 인구/사업체 데이터에 {len(unknown_dongs)}건 포함")
            
        missing_dongs = valid_od_dong_codes - processed_dongs
        if missing_dongs:
            print(f"\nE: OD_dong_list에 있지만 인구/사업체 데이터에는 누락된 동이 {len(missing_dongs)}건")
            
        df = grouped
        
    else:
        df = pdc.check_dong(df, 'dong_code', year)
        
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 결과 저장: {output_path}, 동 개수 : {len(df)}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    # 2023년 데이터 처리
    input_file = "/Users/implement/KT/KTDB/dataset/raw/2021-2023 인구 및 사업자 데이터.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2023.csv"
    process_pop_worker_business(input_file, output_file, '2023')
    
    # 2019년 데이터 처리
    input_file = "/Users/implement/KT/KTDB/dataset/raw/Population Business Worker 2021 2019.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2019.csv"
    process_pop_worker_business(input_file, output_file, '2019')
