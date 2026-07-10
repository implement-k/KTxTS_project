import pandas as pd
import os, json

def process_subway_data(input_path, output_path):
    print(f"processing subway data from {input_path} to {output_path}...")
    df = pd.read_csv(input_path)
    
    # 24, 25, 26년에 생긴 역사 조건 필터링 (26.7.9 나무위키 기준)
    # 형식1: (역사명, 노선_열차종류)
    # 형식2: ("*", 노선_열차종류) -> 노선 일치하는 모든 역사 제외
    # 형식3: (역사명, "*") -> 역사명 일치하는 모든 노선 제외
    exclude_stations = [
        ('*', 'GTXA'),
        ('성남', '경강선'),
        ('암사역사공원', '8호선'),
        ('장자호수공원', '8호선'),
        ('동구릉', '8호선'),
        ('다산', '8호선'),
        ('별내', '8호선'),
        ('구리', '8호선'),
        ('검단호수공원', '인천1호선'),
        ('계양', '인천1호선'),
        ('신검단중앙', '인천1호선'),
        ('*', '과천선'),
        ('*', '안산선'),
        ('*', '안산과천선'),
        ('*', '일산선'),
        ('*', '진접선'),
        ('*', '별내선')
    ]
    
    print(f"Original dataset size: {len(df)}")
    
    # 제외 대상 마스킹
    mask = pd.Series(False, index=df.index)
    for stat, line in exclude_stations:
        if (stat == '*' and line != '*'):
            mask |= (df['노선_열차종류'] == line)
        elif (stat != '*' and line == '*'):
            mask |= (df['철도역명'].str.contains(stat, na=False))
        else:
            mask |= (df['철도역명'].str.contains(stat, na=False)) & (df['노선_열차종류'] == line)
        
    df_filtered = df[~mask].copy()
    print(f"필터 처리된 데이터셋 크기: {len(df_filtered)}")
    
    # 각 역마다 500m 이내 행정동 목록 파싱
    dong_station_count = {}
    
    for _, row in df_filtered.iterrows():
        # 결측치 처리
        if pd.isna(row['행정동코드_500m']):
            print(f"E: 행정동명,코드 존재하지 않음: '{row['철도역명']}' (대표ID: {row['대표ID']}). skip")
            continue
            
        # 행정동코드_500m 파싱 (예: "11240650, 11240660")
        dongs_str = str(row['행정동코드_500m']).strip('[]')
        
        if not dongs_str:
            print(f"E: 행정동코드 파싱 실패: '{row['철도역명']}' (대표ID: {row['대표ID']}). skip")
            continue
            
        dongs = [x.strip() for x in dongs_str.split(',')]
        station_id = row['철도역명'] + '_' + row['노선_열차종류']
        
        if (row['구분'] not in dong_station_count):
            dong_station_count[row['구분']] = {}
        
        for dong_code in dongs:
            if dong_code not in dong_station_count[row['구분']]:
                dong_station_count[row['구분']][dong_code] = set()
            
            dong_station_count[row['구분']][dong_code].add(station_id)
            
            
    # 중간 과정 저장
    json_dict = {
        type: {dong: list(stations) for dong, stations in dong_dict.items()}
        for type, dong_dict in dong_station_count.items()
    }
    
    os.makedirs(os.path.dirname(output_path.replace('.csv', f'_intermediate.json')), exist_ok=True)
    with open(output_path.replace('.csv', '_intermediate.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
   
    # 데이터프레임 변환
    for type, dong_dict in dong_station_count.items():
        dong_station_list = []
        for dong_code, station_ids in dong_dict.items():
            dong_station_list.append({'dong_code': dong_code, 'station_count': len(station_ids)})
        result = pd.DataFrame(dong_station_list)
        
        # 결과 저장
        os.makedirs(os.path.dirname(output_path.replace('.csv', f'_{type}.csv')), exist_ok=True)
        result.to_csv(output_path.replace('.csv', f'_{type}.csv'), index=False, encoding='utf-8-sig')
        print(f"처리 완료. type: {type}, 행정동 수: {len(result)}, 결과 저장: {output_path.replace('.csv', f'_{type}.csv')}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count.csv"
    process_subway_data(input_file, output_file)
