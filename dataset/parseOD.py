import os
import pandas as pd
import geopandas as gpd
import numpy as np

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '../..')
    ktdb_dir = os.path.join(base_dir, 'KTDB')
    os.makedirs(os.path.join(ktdb_dir, 'dataset'), exist_ok=True)
    
    print("Loading 1153 Zone indices...")
    # 1. Load dong_code.xlsx to get the 1153 7-digit codes and define indices
    dong_file = os.path.join(base_dir, 'dataset', 'dong_code.xlsx')
    dong_df = pd.read_excel(dong_file)
    dong_codes_7 = dong_df['읍면동'].astype(str).values
    idx_map = {code: i for i, code in enumerate(dong_codes_7)}
    
    print("Parsing OD data...")
    # 3. Parse ODTRIP23_F.OUT
    od_file = os.path.join(base_dir, 'dataset', '2024-OD-PSN-OBJ-01 수도권 목적OD(2023-2050)', '4. OD', 'ODTRIP23_F.OUT')
    col_names = ['o_taz', 'o_code', 'd_taz', 'd_code', 'm1', 'm2', 'm3', 'm4', 'm5']
    df = pd.read_csv(od_file, sep='\s+', header=None, names=col_names)
    
    # 4. Initialize OD Matrices
    num_dongs = len(dong_codes_7) # Should be 1153
    # X_OD_3D: (1153, 1153, 5)
    X_OD_3D = np.zeros((num_dongs, num_dongs, 5), dtype=np.float32)
    # X_OD_2D: (1153, 1153)
    X_OD_2D = np.zeros((num_dongs, num_dongs), dtype=np.float32)
    
    # 5. Populate matrices
    # Map 7 digit to 0~1152 index
    df['o_idx'] = df['o_code'].astype(str).map(idx_map)
    df['d_idx'] = df['d_code'].astype(str).map(idx_map)
    
    # Drop rows that couldn't be mapped to the 1153 regions (e.g. external zones)
    valid_df = df.dropna(subset=['o_idx', 'd_idx']).copy()
    valid_df['o_idx'] = valid_df['o_idx'].astype(int)
    valid_df['d_idx'] = valid_df['d_idx'].astype(int)
    
    print(f"Valid mapped rows: {len(valid_df)} out of {len(df)}")
    
    # Fill the 3D tensor
    # Purposes: m1(귀가), m2(출근), m3(등교), m4(업무), m5(기타)
    for obj_idx, col in enumerate(['m1', 'm2', 'm3', 'm4', 'm5']):
        # Group by (o_idx, d_idx) and sum
        grouped = valid_df.groupby(['o_idx', 'd_idx'])[col].sum().reset_index()
        for _, row in grouped.iterrows():
            X_OD_3D[int(row['o_idx']), int(row['d_idx']), obj_idx] += row[col]
            
    # Fill the 2D tensor
    valid_df['total'] = valid_df[['m1', 'm2', 'm3', 'm4', 'm5']].sum(axis=1)
    grouped_2d = valid_df.groupby(['o_idx', 'd_idx'])['total'].sum().reset_index()
    for _, row in grouped_2d.iterrows():
        X_OD_2D[int(row['o_idx']), int(row['d_idx'])] += row['total']
        
    out_3d_path = os.path.join(ktdb_dir, 'dataset', 'X_OD_3D.npy')
    out_2d_path = os.path.join(ktdb_dir, 'dataset', 'X_OD_2D.npy')
    
    np.save(out_3d_path, X_OD_3D)
    np.save(out_2d_path, X_OD_2D)
    print(f"✅ Saved X_OD_3D to {out_3d_path} with shape {X_OD_3D.shape}")
    print(f"✅ Saved X_OD_2D to {out_2d_path} with shape {X_OD_2D.shape}")

if __name__ == '__main__':
    main()
