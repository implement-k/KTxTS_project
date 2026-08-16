import json
import torch
from mae_backend_adapter.newtown_predictor import NewtownPreprocessor
from mae_backend_adapter.predictor import MAEPredictor

def main():
    print("1. Preparing Gyosan (교산) inputs...")
    preprocessor = NewtownPreprocessor(newtown_name="gyosan", period="final")
    inputs = preprocessor.prepare()
    
    print(f"   - Total Nodes: {len(inputs.city_codes)}")
    print(f"   - Newtown Zone Codes: {inputs.newtown_zone_codes}")
    
    print("\n2. Initializing MAEPredictor...")
    predictor = MAEPredictor(device="cpu")
    
    print("\n3. Running Prediction (Adapter) ...")
    # predict_from_tensors는 실제 백엔드 요청/응답 구조(JSON 호환 dict)를 반환합니다.
    result = predictor.predict_from_tensors(inputs, request_metadata={"task": 0})
    
    print("\n================ [ Adapter Output Results ] ================")
    od_list = result.get('od', [])
    print(f"Number of OD pairs: {len(od_list)}")
    
    if len(od_list) > 0:
        # Sort by predicted_trips in descending order
        od_list_sorted = sorted(od_list, key=lambda x: x.get('predicted_trips', 0), reverse=True)
        
        print(f"\nTop 30 OD pairs:")
        for i, od in enumerate(od_list_sorted[:30]):
            print(f"  {i+1:2d}. {od['origin_code']} -> {od['destination_code']}: {od['predicted_trips']:,.2f} trips ({od.get('movement_type', '')})")
        
        # Calculate total predicted trips
        total_trips = sum(item.get('trips', 0) for item in od_list)
        print(f"\nTotal Predicted Trips: {total_trips:,.2f}")
    
    metadata = result.get('metadata', {})
    explain = result.get('explainability', {})
    print("\n================ [ Explainability (XAI) Output ] ================")
    print("Keys available:", list(explain.keys()))
    
    if 'error' in explain:
        print(f"Error in XAI: {explain['error']}")
        print(explain.get('traceback', ''))
    elif "feature_importance" in explain:
        fi = explain["feature_importance"]
        print(f"\n- Feature Importance (총 {len(fi)}개 특성)")
        if len(fi) > 0:
            first_key = list(fi.keys())[0]
            if len(fi[first_key]) > 0:
                print(f"  Top feature for {first_key}:", fi[first_key][0])
            
    zones = result.get('newtown_zone_codes', [])
    if "mobility_summary" in explain and len(zones) > 0:
        ms = explain["mobility_summary"].get(zones[0], {})
        if ms:
            print(f"\n- Mobility Summary for {zones[0]}:")
            print(f"  total_inflow: {ms.get('total_inflow', 0):,.2f}")
            print(f"  total_outflow: {ms.get('total_outflow', 0):,.2f}")
            print(f"  net_mobility: {ms.get('net_mobility', 0):,.2f}")
        
    if "od_similarity" in explain:
        ods = explain["od_similarity"]
        sim_nodes = ods.get('similar_nodes', [])
        print(f"\n- OD Similarity (유사한 동네 찾기): {len(sim_nodes)} nodes")
        if len(sim_nodes) > 0:
            print("  Top similar node:", sim_nodes[0])
            
    if "prediction_uncertainty" in explain:
        pu = explain["prediction_uncertainty"]
        print(f"\n- Prediction Uncertainty (불확실성 및 신뢰구간):")
        print(f"  confidence_score: {pu.get('confidence_score')}")
        print(f"  confidence_interval keys: {list(pu.get('confidence_interval', {}).keys())}")
        
    print("\n================================================================")

if __name__ == "__main__":
    main()
