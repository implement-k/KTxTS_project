import pandas as pd
import pickle

base = "mae_backend_adapter/newtown/gyosan"

# 1. OD list
df_od = pd.read_excel(f"{base}/OD_dong_list_2023.xlsx")
print(f"OD dong list shape: {df_od.shape}, columns: {list(df_od.columns)}")

# 2. distance
df_dist = pd.read_csv(f"{base}/dong_distance.csv", index_col=0)
print(f"Distance shape: {df_dist.shape}")

# 3. adjacency
with open(f"{base}/dong_adjacency.pkl", "rb") as f:
    adj = pickle.load(f)
print(f"Adjacency type: {type(adj)}")
if isinstance(adj, pd.DataFrame):
    print(f"Adjacency shape: {adj.shape}")

# 4. static features
df_static = pd.read_csv(f"{base}/newtown_static_features_gyosan.csv")
print(f"Static features shape: {df_static.shape}, columns: {list(df_static.columns)}")
