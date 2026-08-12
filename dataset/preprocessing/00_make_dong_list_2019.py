import pandas as pd
import os

'''
    2019년 데이터 처리 시 제일 먼저 실행해야 함.
'''

def make_dong_list_2019():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, "raw", "dong","OD_dong_list_2019_mismatch.xlsx")
    report_path = os.path.join(base_dir, "raw", "dong", "mismatch_report_2019.xlsx")
    output_path = os.path.join(base_dir, "raw", "dong", "OD_dong_list_2019.xlsx")
    
    print(f"Reading original OD list from {orig_path} to preserve real 10-digit codes...")
    orig_df = pd.read_excel(orig_path)
    orig_map = dict(zip(pd.to_numeric(orig_df['dong_code_10'], errors='coerce'), orig_df['dong_code_10']))
    
    print(f"Reading mismatch report from {report_path}...")
    df = pd.read_excel(report_path)
    
    mapping_records = []
    rep_dong_set:set[int] = set()
    
    for _, row in df.iterrows():
        od_val = row.get('OD데이터')
        if pd.isna(od_val): continue
        
        # 대표 동 코드는 수도권 행정동 용지 비율 2019 데이터로 결정    
        od_code = int(od_val)
        orig_name = str(row.get('동이름', 'Unknown'))
        static_code_str = str(row.get('수도권_행정동_용지_비율_2019', 'nan'))
        
        new_code = od_code # Default to original if no change

        if static_code_str != 'nan':
            clean_code_str = static_code_str.replace('신', '')
            if '/' in clean_code_str:
                new_code = int(float(clean_code_str.split('/')[0]))
            else:
                new_code = int(float(clean_code_str))
                
        code_10 = orig_map.get(od_code, pd.NA)
        rep_dong_set.add(new_code)
                
        mapping_records.append({
            'dong_code_10': od_code,        
            'dong_name': orig_name,
            'dong_code': new_code,          # 대표 8자리 코드
            'station_code_10': code_10      # station 코드
        })

    mapping_df = pd.DataFrame(mapping_records)
    
    # Save the mapping list
    mapping_df.to_excel(output_path, index=False)
    print(f"Successfully generated mapping list: {output_path}")
    print(f"Total mapped codes: {len(mapping_df)}")
    print(f"2019 데이터 최종 동 개수: {len(rep_dong_set)}")

if __name__ == "__main__":
    make_dong_list_2019()