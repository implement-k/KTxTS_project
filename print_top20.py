import pandas as pd
import json
from mae_backend_adapter.newtown_predictor import NewtownPreprocessor
from mae_backend_adapter.predictor import MAEPredictor

print("1. Preparing Gyosan inputs...")
preprocessor = NewtownPreprocessor(newtown_name="gyosan", period="final")
inputs = preprocessor.prepare()

print("\n2. Getting Dong names...")
od_df = pd.read_excel("mae_backend_adapter/newtown/gyosan/OD_dong_list_2023.xlsx")
# Convert dong_code to string to ensure matching
code_to_name = {str(row['dong_code']): str(row['dong_name']) for _, row in od_df.iterrows()}

print("\n3. Running Prediction...")
predictor = MAEPredictor(device="cpu")
result = predictor.predict_from_tensors(inputs, request_metadata={"task": 0})

od_list = result.get('od', [])
# `predicted_trips` might be the key, or `trips`?
for item in od_list:
    item['predicted_trips'] = item.get('predicted_trips', item.get('trips', 0))

sorted_od = sorted(od_list, key=lambda x: x['predicted_trips'], reverse=True)

print("\n================ [ Top 20 OD Pairs ] ================")
for i, item in enumerate(sorted_od[:20]):
    o_name = code_to_name.get(item['origin_code'], item['origin_code'])
    d_name = code_to_name.get(item['destination_code'], item['destination_code'])
    trips = item['predicted_trips']
    mtype = item['movement_type']
    print(f"{i+1:2d}. {o_name} -> {d_name} | {trips:7.2f}명 ({mtype})")
