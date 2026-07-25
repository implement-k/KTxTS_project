import pandas as pd
import os

'''
    2019년 데이터 처리 시 제일 먼저 실행해야 함.
'''

def make_dong_list_2019():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base_dir, "raw", "dong", "mismatch_report_2019.xlsx")
    output_path = os.path.join(base_dir, "raw", "OD_dong_list_2019.xlsx")
    
    print(f"Reading mismatch report from {report_path}...")
    df = pd.read_excel(report_path)
    
    mapping_records = []
    
    for _, row in df.iterrows():
        od_val = row.get('OD데이터')
        if pd.isna(od_val): continue
            
        od_code = int(od_val)
        orig_name = str(row.get('동이름', 'Unknown'))
        note = str(row.get('Unnamed: 5', '')).strip()
        static_code_str = str(row.get('수도권_행정동_용지_비율_2019', 'nan'))
        
        new_code = od_code # Default to original if no change
        
        # Determine the new code based on the mismatch report
        if static_code_str != 'nan':
            # 신 행정동 코드가 포함된 경우, '신'을 제거하고 대표 코드를 선택
            clean_code_str = static_code_str.replace('신', '')
            
            # /가 포함된 경우, 대표 코드를 선택
            if '/' in clean_code_str:
                rep_code = int(float(clean_code_str.split('/')[0]))
                new_code = rep_code
            else:
                new_code = int(float(clean_code_str))
                
        mapping_records.append({
            'dong_code_10': od_code,        # Original OD data dong code
            'dong_name_orig': orig_name,
            'dong_code': new_code,          # Changed code (Representative 8-digit)
            'note': note                    # Modification history
        })

    mapping_df = pd.DataFrame(mapping_records)
    
    # Save the mapping list
    mapping_df.to_excel(output_path, index=False)
    print(f"Successfully generated mapping list: {output_path}")
    print(f"Total mapped codes: {len(mapping_df)}")
    
if __name__ == "__main__":
    make_dong_list_2019()