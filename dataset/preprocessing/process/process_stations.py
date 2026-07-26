import pandas as pd
import os, json

def process_subway_data(input_path, output_path, year = 2023):
    print(f"processing subway data from {input_path} to {output_path}...")
    station_df = pd.read_csv(input_path)
    
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
        ('*', '별내선'),
        ('판교역', 'KTX-이음'),
        ('판교역', '무궁화'),
        ('판교역', '새마을'),
        ('연천역', '1호선'),
        ('청산역', '1호선'),
        ('전곡역', '1호선'),
    ]
    if year == 2019:
        exclude_stations += [
            ('신사역', '신분당선'),
            ('*', '신림선'),
            ('*', '진접선'),
            ('강일역', '5호선'),
            ('미사역', '5호선'),
            ('하남풍산역', '5호선'),
            ('하남시청역', '5호선'),
            ('하남검단산역', '5호선'),
        ]
        
    print(f"Original dataset size: {len(station_df)}")
    
    # 제외 대상 마스킹
    mask = pd.Series(False, index=station_df.index)
    for stat, line in exclude_stations:
        if (stat == '*' and line != '*'):
            mask |= (station_df['노선_열차종류'] == line)
        elif (stat != '*' and line == '*'):
            mask |= (station_df['철도역명'].str.contains(stat, na=False))
        else:
            mask |= (station_df['철도역명'].str.contains(stat, na=False)) & (station_df['노선_열차종류'] == line)
        
    station_df = station_df[~mask].copy()
    print(f"필터 처리된 데이터셋 크기: {len(station_df)}")
    
    dong_station_count: dict[str, dict[str, set[str]]] = {}

    for _, row in station_df.iterrows():
        # 결측치 처리
        if pd.isna(row['행정동코드_500m']):
            print(f"E: 행정동명,코드 존재하지 않음: '{row['철도역명']}' (대표ID: {row['대표ID']}). skip")
            continue
            
        # 행정동코드_500m 파싱 (예: "11240650, 11240660")
        dongs_str: str = str(row['행정동코드_500m']).strip('[]')
        
        if not dongs_str:
            print(f"E: 행정동코드 파싱 실패: '{row['철도역명']}' (대표ID: {row['대표ID']}). skip")
            continue
            
        dongs: list[str] = [x.strip() for x in dongs_str.split(',')]
        
        # 2019년인 경우 10자리 동코드를 매핑 테이블을 참고해 8자리 대표 코드로 변환
        if year == 2019:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            mapping_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", "OD_dong_list_2019.xlsx"))
            
            name_to_code = {}
            for _, map_row in mapping_df.iterrows():
                name = str(map_row['dong_name']).strip()
                if name != 'nan':
                    name_to_code[name] = int(map_row['dong_code'])
                    
            mapping_dict = {}
            names_str = str(row['행정동코드명_500m']).strip('[]')
            if names_str and names_str != 'nan':
                for item in names_str.split(','):
                    if ':' in item:
                        code_str, full_name = item.split(':')
                        try:
                            code_10 = int(code_str.strip())
                            dong_name = full_name.strip().replace("'", "").replace('"', '').split()[-1]
                            
                            if dong_name in name_to_code:
                                mapping_dict[code_10] = name_to_code[dong_name]
                            elif '제' in dong_name and dong_name.replace('제', '') in name_to_code:
                                mapping_dict[code_10] = name_to_code[dong_name.replace('제', '')]
                        except ValueError:
                            pass
                            
            mapped_dongs: list[str] = []
            for d in dongs:
                try:
                    code_10 = int(d.strip())
                    if code_10 in mapping_dict and pd.notna(mapping_dict[code_10]):
                        mapped_dongs.append(str(int(mapping_dict[code_10])))
                except ValueError:
                    pass
            dongs = list(set(mapped_dongs))
            
            # 2019년 데이터에서 매핑 후에도 dongs가 비어있으면 해당 역은 제외
            if not dongs: 
                print(f'W: 해당역 주변에는 행정동이 없음. : {row["철도역명"]}, {row["노선_열차종류"]}. skip')
                continue
            
        station_id: str = row['철도역명'] + '_' + row['노선_열차종류']
        
        if (row['구분'] not in dong_station_count):
            dong_station_count[row['구분']] = {}
        
        for dong_code in dongs:
            if dong_code not in dong_station_count[row['구분']]:
                dong_station_count[row['구분']][dong_code] = set()
            
            dong_station_count[row['구분']][dong_code].add(station_id)
            
            
    # 중간 과정 저장
    json_dict: dict[str, dict[str, list[str]]] = {
        type: {dong: list(stations) for dong, stations in dong_dict.items()}
        for type, dong_dict in dong_station_count.items()
    }
    
    os.makedirs(os.path.dirname(output_path.replace('.csv', f'_intermediate.json')), exist_ok=True)
    with open(output_path.replace('.csv', '_intermediate.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
   
    # 데이터프레임 변환
    for type, dong_dict in dong_station_count.items():
        dong_station_list: list[dict[str, str | int]] = []
        for dong_code, station_id_set in dong_dict.items():
            dong_station_list.append({'dong_code': dong_code, 'station_count': len(station_id_set)})
        result = pd.DataFrame(dong_station_list)
        
        # 결과 저장
        os.makedirs(os.path.dirname(output_path.replace('.csv', f'_{type}.csv')), exist_ok=True)
        result.to_csv(output_path.replace('.csv', f'_{type}.csv'), index=False, encoding='utf-8-sig')
        print(f"처리 완료. type: {type}, 행정동 수: {len(result)}, 결과 저장: {output_path.replace('.csv', f'_{type}.csv')}")

    print("모든 처리 완료.")
    
if __name__ == "__main__":
    # 2019년 데이터 처리
    input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2019.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2019.csv"
    
    # 2023년 데이터 처리
    # input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2023.csv"
    # output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2023.csv"
    process_subway_data(input_file, output_file, year=2019)
