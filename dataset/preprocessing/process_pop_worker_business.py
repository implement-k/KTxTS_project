import pandas as pd
import os
import process_dong_code as pdc

def process_pop_worker_business(input_path, output_path):
    print(f"processing population worker business data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)        
    df = df.drop(columns=['population_year', 'business_year', 'sigungu', 'sido', 'dong_name'])
    
    # 모든 행정동 있는 것 확인 했으므로 그냥 진행
    df = pdc.check_dong(df, 'dong_code')
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료. 결과 저장: {output_path}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    input_file = "/Users/implement/KT/KTDB/dataset/raw/2021-2023 인구 및 사업자 데이터.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count.csv"
    process_pop_worker_business(input_file, output_file)
