import pandas as pd
df = pd.read_csv("mae_backend_adapter/newtown/od_data_2023.csv", index_col=0)
print(f"Shape: {df.shape}")
print(f"Index head: {df.index[:5].tolist()}")
print(f"Columns head: {df.columns[:5].tolist()}")
