import os
import pandas as pd
from config import DATA_DIR

RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)
regions = ['jeju', 'busan', 'daegu', 'daejeon', 'gwangju']

def prep_regional_od():
    for region in regions:
        od_path = os.path.join(RAW_DIR, f'od_{region}.xlsx')
        if not os.path.exists(od_path):
            print(f"Skipping {region}, file not found: {od_path}")
            continue
            
        out_path = os.path.join(PROCESSED_DIR, f'od_{region}.csv')
        if os.path.exists(out_path):
            print(f"Already exists: {out_path}")
            continue
            
        print(f"Loading {region} OD from {od_path}...")
        try:
            # OD Matrix loads '2023년' sheet or '2023' sheet
            xl = pd.ExcelFile(od_path)
            sheet_name = '2023년' if '2023년' in xl.sheet_names else '2023'
            od_df = pd.read_excel(od_path, sheet_name=sheet_name)
            
            # 엑셀 시트에 컬럼명이 '출발', '도착', '귀가', '출근', '등교', '업무', '기타' 등인지 확인
            print(f"Columns for {region}: {list(od_df.columns)}")
            
            # Save to CSV
            od_df.to_csv(out_path, index=False)
            print(f"Saved {out_path}")
            
        except Exception as e:
            print(f"Error processing {region}: {e}")

if __name__ == '__main__':
    prep_regional_od()
