import pandas as pd
import os
import process_dong_code as pdc

def process_land_ratio(input_path, output_path):
    print(f"processing land ratio data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)        
    df = df.drop(columns=['자료권역', '시도', '행정동명', '상업업무지역면적_m2', '공공시설지역면적_m2', '주거지역면적_m2'])
    
    # 일부 행정동 결측
    df = pdc.check_dong(df, '행정동코드')
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 결과 저장: {output_path}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    input_file = "/Users/implement/KT/KTDB/dataset/raw/수도권 행정동 상업 공공 주거 비율.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_land_ratio.csv"
    process_land_ratio(input_file, output_file)
