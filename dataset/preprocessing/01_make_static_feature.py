import pandas as pd
import glob
import os
import sys

# 프로젝트 루트 디렉토리를 경로에 추가하여 import 에러 방지
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dataset.preprocessing.process.process_apartment_ratio as par
import dataset.preprocessing.process.process_land_ratio as plr
import dataset.preprocessing.process.process_pop_worker_business as pwb
import dataset.preprocessing.process.process_stations as ps


def make_static_feature(isFull='n', year='2023'):
    base_dir = "/Users/implement/KT/KTDB/dataset"
    processed_dir = os.path.join(base_dir, "processed")
    
    if year == '2019':
        od_dong_path = os.path.join(base_dir, "raw", "dong", "OD_dong_list_2019.xlsx")
        output_path = os.path.join(processed_dir, "final_static_features_2019.csv")
    else:
        od_dong_path = os.path.join(base_dir, "raw", "dong", "OD_dong_list_2023.xlsx")
        output_path = os.path.join(processed_dir, "final_static_features_2023.csv")
    
    # isFull이면 기존 파일 생성
    if (isFull == 'y'):
        if year == '2023':
            ps_input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2023.csv"
            ps_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2023.csv"
            par_input_file = "/Users/implement/KT/KTDB/dataset/raw/서울 인천 경기 아파트 비율 2024.csv"
            par_output_file = "/Users/implement/KT/KTDB/dataset/processed/processed_apartment_ratio_2023.csv"
            plr_input_file = "/Users/implement/KT/KTDB/dataset/raw/수도권 행정동 상업 공공 주거 비율.csv"
            plr_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_land_ratio_2023.csv"
            pwb_input_file = "/Users/implement/KT/KTDB/dataset/raw/2021-2023 인구 및 사업자 데이터.csv"
            pwb_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2023.csv"
            ps.process_subway_data(ps_input_file, ps_output_file, 2023) # 2023
        elif year == '2019':
            ps_input_file = "/Users/implement/KT/KTDB/dataset/raw/Station Line Admin Dataset_2019.csv" # TODO 2019년도 데이터 오면 수정 해야함.
            ps_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_subway_count_2019.csv" #2019
            par_input_file = "/Users/implement/KT/KTDB/dataset/raw/서울 인천 경기 아파트 비율 2019.csv" # 2019
            par_output_file = "/Users/implement/KT/KTDB/dataset/processed/processed_apartment_ratio_2019.csv" # 2019
            plr_input_file = "/Users/implement/KT/KTDB/dataset/raw/수도권 행정동 용지 비율 2019.csv" # 2019
            plr_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_land_ratio_2019.csv" #2019
            pwb_input_file = "/Users/implement/KT/KTDB/dataset/raw/Population Business Worker 2021 2019.csv" #2019
            pwb_output_file = "/Users/implement/KT/KTDB/dataset/processed/dong_pop_worker_business_count_2019.csv" #2019
            ps.process_subway_data(ps_input_file, ps_output_file, 2019) # 2019
        
        par.process_apartment_ratio(par_input_file, par_output_file, year)
        plr.process_land_ratio(plr_input_file, plr_output_file, year)
        pwb.process_pop_worker_business(pwb_input_file, pwb_output_file)
        
    
    print("OD 동 불러오기")
    base_df = pd.read_excel(od_dong_path)
    if 'dong_code' not in base_df.columns:
        raise ValueError("OD_dong_list.xlsx에 'dong_code' 컬럼이 없습니다.")
    
    # 병합할 데이터프레임 초기화
    if 'dong_name_orig' in base_df.columns and 'dong_name' not in base_df.columns:
        base_df.rename(columns={'dong_name_orig': 'dong_name'}, inplace=True)
        
    merged_df = base_df[['dong_code', 'dong_name']].copy()
    merged_df.drop_duplicates(subset=['dong_code'], keep='first', inplace=True)
    
    # processed 폴더의 모든 csv 파일 병합 (년도에 따라 필터링)
    all_csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if year == '2019':
        csv_files = [f for f in all_csv_files if "_2019" in os.path.basename(f) and "final_static_features" not in os.path.basename(f)]
    else:
        csv_files = [f for f in all_csv_files if "_2023" in os.path.basename(f) and "final_static_features" not in os.path.basename(f)]
    
    subway_columns = []
    other_feature_columns = []
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n[{file_name}] 병합 처리 중...")
        
        df = pd.read_csv(file_path)
        
        # 행정구역코드 또는 행정동코드 이름을 dong_code로 통일
        if '행정구역코드' in df.columns:
            df.rename(columns={'행정구역코드': 'dong_code'}, inplace=True)
        if '행정동코드' in df.columns:
            df.rename(columns={'행정동코드': 'dong_code'}, inplace=True)
            
        if 'dong_code' not in df.columns:
            print(f"  -> 경고: {file_name}에 dong_code(또는 행정구역코드)가 없습니다. 스킵합니다.")
            continue
            
        # 지하철(철도) 데이터 처리
        if file_name.startswith("dong_subway_count_"):
            # 파일명에서 철도 타입 추출 (예: dong_subway_count_지하철.csv -> 지하철)
            base_name = file_name.replace(".csv", "").replace("dong_subway_count_", "")
            if base_name.startswith("2019_"):
                subway_type = base_name.replace("2019_", "", 1)
            else:
                subway_type = base_name
                
            new_col_name = f"station_count_{subway_type}"
            
            if 'station_count' in df.columns:
                df.rename(columns={'station_count': new_col_name}, inplace=True)
                subway_columns.append(new_col_name)
            
            # 병합
            df = df[['dong_code', new_col_name]]
            merged_df = pd.merge(merged_df, df, on='dong_code', how='left')
            
        # 아파트 비율 데이터 처리
        elif "apartment_ratio" in file_name:
            cols_to_keep = ['dong_code', '아파트비율_퍼센트']
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[cols_to_keep]
            merged_df = pd.merge(merged_df, df, on='dong_code', how='left')
            if '아파트비율_퍼센트' in df.columns:
                other_feature_columns.append('아파트비율_퍼센트')
                
        # land_ratio 데이터 처리
        elif "land_ratio" in file_name:
            # 2019와 2023의 컬럼명이 달라서 통일
            rename_dict = {
                '행정동총면적_m2': '행정동전체면적_m2',
                '주거용지비율': '주거지역비율_pct',
                '상업업무용지비율': '상업업무지역비율_pct',
                '공공시설용지비율': '공공시설지역비율_pct'
            }
            df.rename(columns=rename_dict, inplace=True)
            
            # 불필요한 컬럼 제거 (행정동명, 시도, 자료권역 등)
            drop_cols = [c for c in ['행정동명', '시도', '자료권역'] if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)
            
            feature_cols = [c for c in df.columns if c != 'dong_code']
            merged_df = pd.merge(merged_df, df, on='dong_code', how='left')
            other_feature_columns.extend(feature_cols)
                
        # 인구/종사자/사업체 데이터 처리
        elif "pop_worker_business" in file_name:
            # dong_code 제외한 나머지 컬럼 추출
            feature_cols = [c for c in df.columns if c != 'dong_code']
            merged_df = pd.merge(merged_df, df, on='dong_code', how='left')
            other_feature_columns.extend(feature_cols)
            
        else:
            print(f"  -> 참고: {file_name}은 병합 규칙이 명확하지 않아 병합하지 않습니다.")
    
    # 3. 결측치(NaN) 처리 로직
    print("\n결측치(NaN) 채우기 작업 시작...")
    
    # 철도 데이터는 매칭되지 않으면 무조건 0으로 채움
    for col in subway_columns:
        if col in merged_df.columns:
            missing_cnt = merged_df[col].isna().sum()
            merged_df[col] = merged_df[col].fillna(0).astype(int)
            print(f"  - {col}: {missing_cnt}건의 누락값을 0으로 채웠습니다.")
            
    # 나머지 특성은 평균으로 채움
    # 시군구 코드 (앞 5자리)와 시도 코드 (앞 2자리) 추출
    merged_df['sigungu_code'] = merged_df['dong_code'] // 1000
    merged_df['sido_code'] = merged_df['dong_code'] // 1000000
    
    for col in other_feature_columns:
        if col in merged_df.columns:
            missing_cnt = merged_df[col].isna().sum()
            if missing_cnt > 0:
                # 시군구 평균값으로 먼저 채우기
                sigungu_mean = merged_df.groupby('sigungu_code')[col].transform('mean')
                merged_df[col] = merged_df[col].fillna(sigungu_mean)
                
                # 시군구 전체가 결측이어서 아직도 누락된 곳은 시도 평균값으로 채우기
                sido_mean = merged_df.groupby('sido_code')[col].transform('mean')
                merged_df[col] = merged_df[col].fillna(sido_mean)
                
                # 소수점 2자리 반올림
                merged_df[col] = merged_df[col].round(2)
                print(f"  - {col}: {missing_cnt}건의 누락값을 시군구/시도 평균값으로 채웠습니다.")
            else:
                print(f"  - {col}: 누락값이 없습니다.")
                
    # 임시 컬럼 제거
    merged_df.drop(columns=['sigungu_code', 'sido_code'], inplace=True)
    rename_map = {
        'station_count_2023_준고속철도': 'station_count_준고속철도',
        'station_count_2023_고속철도': 'station_count_고속철도',
        'station_count_2023_일반철도': 'station_count_일반철도',
        'station_count_2023_지하철': 'station_count_지하철',
    }

    merged_df.rename(columns=rename_map, inplace=True)
    merged_df = merged_df.drop(columns=['시군구'], errors='ignore')
    
    print("추가 파생 변수(밀도, 기타지역비율 등) 생성 중...")
    # 밀도 추가
    if 'worker_count' in merged_df.columns and '행정동전체면적_m2' in merged_df.columns:
        merged_df['worker_density'] = merged_df['worker_count'] / (merged_df['행정동전체면적_m2'] + 1e-5)
    if 'business_count' in merged_df.columns and '행정동전체면적_m2' in merged_df.columns:
        merged_df['business_density'] = merged_df['business_count'] / (merged_df['행정동전체면적_m2'] + 1e-5)
    if 'station_count_지하철' in merged_df.columns and '행정동전체면적_m2' in merged_df.columns:
        merged_df['station_density_지하철'] = merged_df['station_count_지하철'] / (merged_df['행정동전체면적_m2'] + 1e-5)
        
    # 기타지역비율_pct 추가 (비율 총합 100% 맞추기 위함)
    if all(col in merged_df.columns for col in ['상업업무지역비율_pct', '공공시설지역비율_pct', '주거지역비율_pct']):
        merged_df['기타지역비율_pct'] = 100.0 - (merged_df['상업업무지역비율_pct'] + merged_df['공공시설지역비율_pct'] + merged_df['주거지역비율_pct'])
        merged_df['기타지역비율_pct'] = merged_df['기타지역비율_pct'].clip(lower=0.0)
    
    # 결과 저장
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n최종 데이터셋(총 {len(merged_df)}행, {len(merged_df.columns)}열)이 다음 경로에 저장되었습니다:")
    print(f" -> {output_path}")

if __name__ == "__main__":
    isFull = input("각 csv 파일 모두 생성 후 static_feature 생성? (y/n):")   
    if isFull.lower() not in ['y', 'n']:
        print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.") 
    else:
        year = input("년도 선택 (2019/2023):")
        if year not in ['2019', '2023']:
            print("잘못된 입력입니다. '2019' 또는 '2023'을 입력해주세요.")
        else:
            make_static_feature(isFull.lower(), year)
