import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_dong_code as pdc
import numpy as np

def process_land_ratio(input_path, output_path, year='2023'):
    print(f"processing land ratio data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path) 
    try:       
        df = df.drop(columns=['자료권역', '시도', '행정동명', '상업업무지역면적_m2', '공공시설지역면적_m2', '주거지역면적_m2'])
    except:
        pass
    # 2019년의 경우 자체 매핑 및 검증 로직이 있으므로 check_dong은 건너뜁니다.
    
    if year == '2019':
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        od_dong_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
        mismatch_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"mismatch_report_{year}.xlsx"))
        valid_od_dongs = set(od_dong_df['dong_code'])
        od10_to_rep = dict(zip(pd.to_numeric(od_dong_df['dong_code_10'], errors='coerce'), od_dong_df['dong_code']))
        od8_to_rep = dict(zip(pd.to_numeric(od_dong_df['dong_code'], errors='coerce'), od_dong_df['dong_code']))
        od_to_rep = {**od10_to_rep, **od8_to_rep}
        mapping_dict = {}
        
        for _, row in mismatch_df.iterrows():
            od_code = row['OD데이터']
            codes_str = str(row['수도권_행정동_용지_비율_2019'])
            
            if pd.isna(od_code) or codes_str == 'nan': continue
                
            od_code_int = int(float(od_code))
            if od_code_int not in od_to_rep:
                continue
            rep_code = od_to_rep[od_code_int]
            
            tokens = codes_str.split('/')
            for token in tokens:
                token = token.strip()
                new_code_str = token.replace('신', '') if token.startswith('신') else token
                if new_code_str.isdigit():
                    mapping_dict[int(new_code_str)] = rep_code
        
        df['행정동코드'] = pd.to_numeric(df['행정동코드'], errors='coerce')
        df['mapped_code'] = df['행정동코드'].map(lambda x: mapping_dict.get(x, x))
        
        df['주거용지면적_m2'] = df['주거용지비율'] * df['행정동총면적_m2']
        df['상업업무용지면적_m2'] = df['상업업무용지비율'] * df['행정동총면적_m2']
        df['공공시설용지면적_m2'] = df['공공시설용지비율'] * df['행정동총면적_m2']
        
        grouped = df.groupby('mapped_code').agg({
            '행정동명': lambda x: '/'.join(x.unique()),
            '행정동총면적_m2': 'sum',
            '주거용지면적_m2': 'sum',
            '상업업무용지면적_m2': 'sum',
            '공공시설용지면적_m2': 'sum'
        }).reset_index()
        
        total_area = grouped['행정동총면적_m2']
        
        grouped['주거용지비율'] = np.where(total_area > 0, 
                            grouped['주거용지면적_m2'] / total_area, 0)
        grouped['상업업무용지비율'] = np.where(total_area > 0, 
                            grouped['상업업무용지면적_m2'] / total_area, 0)
        grouped['공공시설용지비율'] = np.where(total_area > 0, 
                            grouped['공공시설용지면적_m2'] / total_area, 0)
        
        grouped.rename(columns={'mapped_code': '행정동코드'}, inplace=True)
        grouped.drop(columns=['주거용지면적_m2', '상업업무용지면적_m2', '공공시설용지면적_m2'], inplace=True)
        processed_dongs = set(grouped['행정동코드'])
        
        # 용지 비율 데이터에는 있지만 OD_dong_list에는 없는 동 (미승인/알 수 없는 동)
        unknown_dongs = processed_dongs - valid_od_dongs
        if unknown_dongs:
            print(f"\nE: OD_dong_list에 존재하지 않는 동이 용지 비율 데이터에 {len(unknown_dongs)}건 포함")
            unknown_df = grouped[grouped['행정동코드'].isin(unknown_dongs)]
            for _, row in unknown_df.iterrows():
                print(f"  - 코드: {int(row['행정동코드'])}, 이름: {row['행정동명']}")
        
        # OD_dong_list에는 있지만 용지 비율 데이터에는 없는 동 (부족한 동)
        missing_dongs = valid_od_dongs - processed_dongs
        if missing_dongs:
            print(f"\nE: OD_dong_list에 있지만 용지 비율 데이터에는 누락된 동이 {len(missing_dongs)}건")
            missing_df = od_dong_df[od_dong_df['dong_code'].isin(missing_dongs)]
            for _, row in missing_df.iterrows():
                print(f"  - 누락 코드: {int(row['dong_code'])}, 동이름: {row['dong_name']}")
                
        # 결과 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grouped.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"동 개수: {len(grouped)}")
    else:    
        df = pdc.check_dong(df, '행정동코드', year)
        # 결과 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"동 개수: {len(df)}")
    print(f"처리 완료. 결과 저장: {output_path}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    # 2023
    input_file = "/Users/implement/KT/KTDB/dataset/raw/수도권 행정동 용지 비율.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_land_ratio_2023.csv"
    process_land_ratio(input_file, output_file, '2023')
    
    # 2019
    input_file = "/Users/implement/KT/KTDB/dataset/raw/수도권 행정동 용지 비율 2019.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_land_ratio_2019.csv"
    process_land_ratio(input_file, output_file, '2019')
