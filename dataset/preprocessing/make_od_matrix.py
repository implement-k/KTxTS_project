import pandas as pd
import os

def make_od_matrix():
    input_file = "/Users/implement/KT/KTDB/dataset/raw/ODTRIP23_F.OUT"
    output_file = "/Users/implement/KT/KTDB/dataset/od_data.csv"
    
    print(f"파일 읽기 시작: {input_file}")
    
    columns = [
        'O_index', 'O_dong_code', 
        'D_index', 'D_dong_code', 
        '귀가', '출근', '등교', '업무', '기타'
    ]
    
    # 대용량 데이터이므로 engine='c' 사용
    df = pd.read_csv(input_file, sep=r'\s+', names=columns, engine='c')
    
    print(f"데이터 로드 완료 (총 {len(df):,}행). 행정동 코드 변환 시작.")
    
    # 동 코드 7자리 -> 8자리 표준 변환
    # Origin 
    o_code = pd.to_numeric(df['O_dong_code'], errors='coerce')
    df['O_dong_code'] = o_code.mask(o_code < 10000000, o_code * 10).fillna(0).astype(int)
    
    # Destination 
    d_code = pd.to_numeric(df['D_dong_code'], errors='coerce')
    df['D_dong_code'] = d_code.mask(d_code < 10000000, d_code * 10).fillna(0).astype(int)
    
    
    # 메모리 용량 및 로딩 속도 최적화를 위한 데이터 타입 변경 
    for col in ['귀가', '출근', '등교', '업무', '기타']:
        df[col] = df[col].astype('float32')
    
    for col in ['O_index', 'D_index']:
        df[col] = df[col].astype('int32')
        
    print(f"전처리 최적화 완료")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"성공적으로 저장: {output_file}")


if __name__ == "__main__":
    make_od_matrix()
