import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
from config import STATIC_DATA_PATH

def main():
    if not os.path.exists(STATIC_DATA_PATH):
        print(f"Error: {STATIC_DATA_PATH} not found.")
        return
        
    static_df = pd.read_csv(STATIC_DATA_PATH)
    
    # Check correlation between station_count_지하철 and pop_60_plus
    corr = static_df['station_count_지하철'].corr(static_df['pop_60_plus'])
    print(f"Pearson Correlation between 'station_count_지하철' and 'pop_60_plus': {corr:.4f}")
    
    # Also check with station_density
    static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
    static_df['pop_60_plus_density'] = static_df['pop_60_plus'] / (static_df['행정동전체면적_m2'] + 1e-5)
    
    corr_density = static_df['station_density_지하철'].corr(static_df['pop_60_plus_density'])
    print(f"Pearson Correlation between 'station_density_지하철' and 'pop_60_plus_density': {corr_density:.4f}")
    
    # Additional correlations for context
    print("\n--- Correlation of 'pop_60_plus' with other features ---")
    corr_others = static_df.corr(numeric_only=True)['pop_60_plus'].sort_values(ascending=False)
    print(corr_others.head(5))
    print("...")
    print(corr_others.tail(5))
    
    print("\n--- Correlation of 'station_count_지하철' with other features ---")
    corr_station = static_df.corr(numeric_only=True)['station_count_지하철'].sort_values(ascending=False)
    print(corr_station.head(5))
    print("...")
    print(corr_station.tail(5))

if __name__ == '__main__':
    main()
