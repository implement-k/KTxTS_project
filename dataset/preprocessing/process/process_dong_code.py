import pandas as pd
import os

def get_advanced_map(year='2023'):
    """
    mismatch_report의 note 컬럼을 읽어, 1:N으로 연쇄 분할되는 동들의 매핑 정보(advanced_map)를 생성합니다.
    (예: 송도2동 -> 송도2동, 송도4동 -> 송도4동, 송도5동)
    반환값:
        advanced_map: {old_name: [mapped_name1, mapped_name2, ...]}
        name_to_code: {dong_name: dong_code}
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mapping_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
    
    name_to_code = {}
    for _, map_row in mapping_df.iterrows():
        name = str(map_row['dong_name']).strip()
        if name != 'nan':
            name_to_code[name] = int(map_row['dong_code'])
            
    df_mis = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"mismatch_report_{year}.xlsx"))
    
    rules: dict[str, list[str]] = {}
    note_col = None
    if year == '2019':
        # 2019년은 원래 연쇄 분할을 사용하지 않고 mismatch_report의 각 열 1:1 매핑만으로 누락 없이 처리되었습니다.
        pass
    elif year == '2023':
        if 'Unnamed: 6' in df_mis.columns:
            note_col = 'Unnamed: 6'
            
    if note_col:
        for _, mis_row in df_mis.iterrows():
            if pd.notna(mis_row[note_col]):
                note = str(mis_row[note_col])
                for rule_part in note.split(','):
                    rule_part = rule_part.strip()
                    if '->' in rule_part:
                        parts = rule_part.split('->')
                        if len(parts) >= 2:
                            old_name = parts[0].strip()
                            new_names_str = parts[-1].strip()
                            new_names = [n.strip() for n in new_names_str.split('/') if n.strip()]
                            if old_name and new_names:
                                rules[old_name] = new_names
                                
    target_dongs = set(name_to_code.keys())
    advanced_map = {d: [d] for d in target_dongs}
    for old_name in rules.keys():
        if old_name not in advanced_map:
            advanced_map[old_name] = [old_name]
            
    changed = True
    max_iter = 100
    iter_count = 0
    while changed:
        iter_count += 1
        if iter_count > max_iter:
            print("W: 최대 반복 횟수 초과 — rules에 순환 참조가 있습니다. 매핑을 강제 종료합니다.")
            break
        changed = False
        for d in list(advanced_map.keys()):
            current_targets = advanced_map[d]
            new_targets = []
            for t in current_targets:
                if t in rules:
                    split_targets = rules[t]
                    newly_spawned = set(split_targets) - {t}
                    
                    # Prevent self-loop
                    if set(split_targets) == set([t]):
                        new_targets.append(t)
                        continue
                        
                    # Target dataset에 없는 분할 동이 포함되어 있을 때만 확장
                    if not any(n in target_dongs for n in newly_spawned):
                        new_targets.extend(split_targets)
                        changed = True
                        continue
                new_targets.append(t)
            
            seen = set()
            dedup = []
            for nt in new_targets:
                if nt not in seen:
                    seen.add(nt)
                    dedup.append(nt)
            advanced_map[d] = dedup
            
    return advanced_map, name_to_code


def map_codes_from_mismatch(df, source_col, mismatch_target_col, year='2023'):
    """
    mismatch_report의 특정 컬럼(mismatch_target_col) 정보를 이용하여, 
    df의 source_col에 들어있는 구 코드를 최신 타겟 코드로 1:N 매핑(복제)해줍니다.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    od_dong_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"OD_dong_list_{year}.xlsx"))
    mismatch_df = pd.read_excel(os.path.join(base_dir, "raw", "dong", f"mismatch_report_{year}.xlsx"))
    
    valid_od_dongs = set(od_dong_df['dong_code'])
    
    mapping_dict = {}
    
    od10_to_rep = {}
    if 'dong_code_10' in od_dong_df.columns:
        for k, v in zip(od_dong_df['dong_code_10'], od_dong_df['dong_code']):
            if pd.notna(k): od10_to_rep[int(k)] = int(v)
    od8_to_rep = {int(k): int(v) for k, v in zip(od_dong_df['dong_code'], od_dong_df['dong_code'])}
    od_to_rep = {**od10_to_rep, **od8_to_rep}
    
    advanced_map, name_to_code = get_advanced_map(year)
    
    for _, row in mismatch_df.iterrows():
        od_code_val = row['OD데이터']
        if mismatch_target_col not in mismatch_df.columns:
            continue
        codes_str = str(row[mismatch_target_col])
        dong_name = str(row['동이름']).strip()
        
        if pd.isna(od_code_val) or codes_str == 'nan':
            continue
            
        od_code_int = int(float(od_code_val))
        if od_code_int not in od_to_rep:
            continue
            
        rep_code = od_to_rep[od_code_int]
        target_codes = [rep_code]
        
        if dong_name in advanced_map and set(advanced_map[dong_name]) != {dong_name}:
            mapped_ints = []
            for mapped_name in advanced_map[dong_name]:
                if mapped_name in name_to_code:
                    mapped_ints.append(name_to_code[mapped_name])
                elif mapped_name + '동' in name_to_code:
                    mapped_ints.append(name_to_code[mapped_name + '동'])
            if mapped_ints:
                target_codes = mapped_ints
                
        tokens = codes_str.split('/')
        for token in tokens:
            token = token.strip()
            new_code_str = token.replace('신', '') if token.startswith('신') else token
            if new_code_str.endswith('.0'):
                new_code_str = new_code_str[:-2]
            if new_code_str.isdigit():
                mapping_dict[int(new_code_str)] = target_codes

    df_code = pd.to_numeric(df[source_col], errors='coerce')
    df_code_8digit = df_code.mask(df_code < 10000000, df_code * 10).fillna(0).astype(int)
    df['raw_mapped_code'] = df_code_8digit
    
    expanded_rows = []
    for _, row in df.iterrows():
        raw_code = row['raw_mapped_code']
        targets = mapping_dict.get(raw_code, [raw_code])
        for t in targets:
            new_row = row.copy()
            new_row['mapped_code'] = t
            expanded_rows.append(new_row)
            
    df_expanded = pd.DataFrame(expanded_rows)
    df_expanded.drop(columns=['raw_mapped_code'], inplace=True)
    
    return df_expanded, valid_od_dongs


def check_dong(df, dong_cal_name, year='2023'):
    """ 단순 매핑 없이, valid하지 않은 동을 날리는 용도 (기존 유지) """
    od_dong_list = pd.read_excel(f"/Users/implement/KT/KTDB/dataset/raw/dong/OD_dong_list_{year}.xlsx")
    
    df_code = pd.to_numeric(df[dong_cal_name], errors='coerce')
    df_code_8digit = df_code.mask(df_code < 10000000, df_code * 10)
    
    valid_dongs = set(od_dong_list['dong_code'])
    if 'dong_code_10' in od_dong_list.columns:
        valid_dongs.update(od_dong_list['dong_code_10'].dropna())
    
    invalid_mask = df_code_8digit.isna() | ~df_code_8digit.isin(valid_dongs)
    
    invalid_rows = df[invalid_mask]
    if not invalid_rows.empty:
        invalid_unique_codes = invalid_rows[dong_cal_name].unique()
        print(f"E: 행정동코드 누락 또는 존재하지 않음. 총 {len(invalid_rows)}건 skip 처리됨.")
        print(f"   제외된 코드 목록(원본): {invalid_unique_codes}")
        
    df_filtered = df[~invalid_mask].copy()
    df_filtered[dong_cal_name] = df_code_8digit[~invalid_mask].astype(int)
    
    return df_filtered
