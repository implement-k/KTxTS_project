import pandas as pd
import os, json, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import process_dong_code as pdc

def process_subway_data(input_path, output_path, year = 2023):
    print(f"processing subway data from {input_path} to {output_path}...")
    station_df = pd.read_csv(input_path)
    
    exclude_stations = [
        ('*', 'GTXA'), ('성남', '경강선'), ('암사역사공원', '8호선'),
        ('장자호수공원', '8호선'), ('동구릉', '8호선'), ('다산', '8호선'),
        ('별내', '8호선'), ('구리', '8호선'), ('검단호수공원', '인천1호선'),
        ('계양', '인천1호선'), ('신검단중앙', '인천1호선'), ('*', '과천선'),
        ('*', '안산선'), ('*', '안산과천선'), ('*', '일산선'), ('*', '별내선'),
        ('판교역', 'KTX-이음'), ('판교역', '무궁화'), ('판교역', '새마을'),
        ('연천역', '1호선'), ('청산역', '1호선'), ('전곡역', '1호선'),
    ]
    if year == 2019:
        exclude_stations += [
            ('신사역', '신분당선'), ('*', '신림선'), ('*', '진접선'),
            ('강일역', '5호선'), ('미사역', '5호선'), ('하남풍산역', '5호선'),
            ('하남시청역', '5호선'), ('하남검단산역', '5호선'),
        ]
        
    mask = pd.Series(False, index=station_df.index)
    for stat, line in exclude_stations:
        if (stat == '*' and line != '*'):
            mask |= (station_df['노선_열차종류'] == line)
        elif (stat != '*' and line == '*'):
            mask |= (station_df['철도역명'].str.contains(stat, na=False))
        else:
            mask |= (station_df['철도역명'].str.contains(stat, na=False)) & (station_df['노선_열차종류'] == line)
        
    station_df = station_df[~mask].copy()
    
    dong_station_count: dict[str, dict[str, set[str]]] = {}
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mapping_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
    valid_dongs = set(mapping_df['dong_code'].astype(str))
    
    # name_to_code 
    name_to_code = {}
    od10_to_rep = {}
    if 'dong_code_10' in mapping_df.columns:
        for k, v in zip(mapping_df['dong_code_10'], mapping_df['dong_code']):
            if pd.notna(k): od10_to_rep[int(k)] = int(v)
            
    od8_to_rep = {int(k): int(v) for k, v in zip(mapping_df['dong_code'], mapping_df['dong_code'])}
    od_to_rep = {**od10_to_rep, **od8_to_rep}
    
    advanced_map, name_to_code = pdc.get_advanced_map(year=str(year))
    
    unmapped_station_dongs = set()
    
    # Create code-based advanced map (for 2023 that has no names)
    # Map old codes to new split codes if available
    mismatch_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"mismatch_report_{year}.xlsx"))
    
    code_advanced_map = {}
    for _, row in mismatch_df.iterrows():
        od_code_val = row['OD데이터']
        dong_name = str(row['동이름']).strip()
        if pd.notna(od_code_val):
            od_code_int = int(float(od_code_val))
            if od_code_int in od_to_rep:
                rep_code = od_to_rep[od_code_int]
                target_codes = [rep_code]
                if dong_name in advanced_map:
                    mapped_ints = []
                    for mapped_name in advanced_map[dong_name]:
                        if mapped_name in name_to_code:
                            mapped_ints.append(name_to_code[mapped_name])
                        elif mapped_name + '동' in name_to_code:
                            mapped_ints.append(name_to_code[mapped_name + '동'])
                    if mapped_ints:
                        target_codes = mapped_ints
                # Any code that maps to this rep_code (e.g. 10 digit, old 8 digit)
                # should be split to target_codes
                # In mismatch_report we only know the rep_code. But wait!
                pass # The code_advanced_map is hard to build backward without knowing old codes.

    for _, row in station_df.iterrows():
        if pd.isna(row['행정동코드_500m']):
            continue
            
        dongs_str: str = str(row['행정동코드_500m']).strip('[]')
        if not dongs_str:
            continue
            
        dongs: list[str] = [x.strip() for x in dongs_str.split(',')]
        mapped_dongs: list[str] = []
        unmapped_for_row: list[str] = []
        
        # 2019: Use name based mapping if available
        if year == 2019:
            mapping_dict = {}
            names_str = str(row['행정동코드명_500m']).strip('[]')
            if names_str and names_str != 'nan':
                for item in names_str.split(','):
                    if ':' in item:
                        code_str, full_name = item.split(':')
                        try:
                            code_10 = int(code_str.strip())
                            full_name_clean = full_name.strip().replace("'", "").replace('"', '').replace('용인시처인구', '용인시 처인구')
                            words = full_name_clean.split()
                            
                            dong_name1 = words[-1]
                            dong_name2 = words[-2] + ' ' + words[-1] if len(words) > 1 else dong_name1
                            
                            candidates = [dong_name1, dong_name2]
                            candidates += [c.replace('제', '') for c in candidates if '제' in c]
                            candidates += [c.replace('·', ',') for c in candidates if '·' in c]
                            
                            for c in candidates:
                                if c in advanced_map:
                                    mapped_ints = []
                                    for target_name in advanced_map[c]:
                                        if target_name in name_to_code:
                                            mapped_ints.append(name_to_code[target_name])
                                        elif target_name + '동' in name_to_code:
                                            mapped_ints.append(name_to_code[target_name + '동'])
                                    if mapped_ints:
                                        mapping_dict[code_10] = mapped_ints
                                        break
                        except ValueError:
                            pass
                            
            for d in dongs:
                try:
                    code_10 = int(d.strip())
                    if code_10 in mapping_dict:
                        for target_code in mapping_dict[code_10]:
                            mapped_dongs.append(str(int(target_code)))
                    elif code_10 in od_to_rep:
                        mapped_dongs.append(str(int(od_to_rep[code_10])))
                except ValueError:
                    pass
        else:
            # 2023: Code-based mapping (8 digit)
            for d in dongs:
                try:
                    code_val = int(d.strip())
                    if code_val in od_to_rep:
                        mapped_dongs.append(str(int(od_to_rep[code_val])))
                    else:
                        # Fallback mapping if there are known 8-digit splits
                        # For now, just keep it if it's valid
                        if str(code_val) in valid_dongs:
                            mapped_dongs.append(str(code_val))
                        else:
                            unmapped_for_row.append(d)
                except ValueError:
                    unmapped_for_row.append(d)
                    
        dongs = list(set(mapped_dongs))
        if not dongs: 
            unmapped_station_dongs.update(unmapped_for_row)
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
    
    os.makedirs(os.path.dirname(output_path.replace('.csv', f'_{year}_intermediate.json')), exist_ok=True)
    with open(output_path.replace('.csv', f'_{year}_intermediate.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
   
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    od_dong_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
    valid_dongs = set(od_dong_df['dong_code'].astype(str))

    for type, dong_dict in dong_station_count.items():
        dong_station_list = []
        for dong_code, station_id_set in dong_dict.items():
            dong_station_list.append({'dong_code': dong_code, 'station_count': len(station_id_set)})
        result = pd.DataFrame(dong_station_list)
        
        if not result.empty:
            missing_dongs = valid_dongs - set(result['dong_code'])
            if missing_dongs:
                print(f"I: OD_dong_list에 있지만 {type} 데이터에 누락된(역이 없는) 동이 {len(missing_dongs)}건. 0으로 패딩 처리됩니다.")
        else:
            print(f"I: {type} 데이터에 매핑된 동이 없습니다.")

        type_output = output_path.replace('.csv', f'_{type}.csv')
        os.makedirs(os.path.dirname(type_output), exist_ok=True)
        result.to_csv(type_output, index=False, encoding='utf-8-sig')

    if unmapped_station_dongs:
        print(f"W: 역 데이터에 존재하지만 OD_dong_list에 매핑할 수 없는 동 코드(또는 이름)가 {len(unmapped_station_dongs)}건 있습니다.")
        print(f"   목록: {unmapped_station_dongs}")

if __name__ == "__main__":
    input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2019.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2019.csv"
    process_subway_data(input_file, output_file, year=2019)
    
    input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2023.csv"
    output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2023.csv"
    process_subway_data(input_file, output_file, year=2023)
