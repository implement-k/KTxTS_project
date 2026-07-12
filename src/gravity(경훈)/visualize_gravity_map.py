import importlib.util
import json
import os
import sys

import folium
import numpy as np
import pandas as pd
from shapely.geometry import shape


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
GRAVITY_DIR = os.path.join(SRC_DIR, "gravity(경훈)")
GEOJSON_PATH = os.path.join(ROOT_DIR, "dataset", "raw", "dong", "dong_area_20230101.geojson")
DONG_LIST_PATH = os.path.join(ROOT_DIR, "dataset", "raw", "OD_dong_list.xlsx")
MAP_OUTPUT_PATH = os.path.join(ROOT_DIR, "gravity_prediction_map.html")
CSV_OUTPUT_PATH = os.path.join(ROOT_DIR, "gravity_prediction_top_flows.csv")
TOTALS_OUTPUT_PATH = os.path.join(ROOT_DIR, "gravity_prediction_test_totals.csv")

# This script now lives inside src/gravity(경훈).
# Recompute paths from the script location so it works from any working directory.
GRAVITY_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(GRAVITY_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
GEOJSON_PATH = os.path.join(ROOT_DIR, "dataset", "raw", "dong", "dong_area_20230101.geojson")
DONG_LIST_PATH = os.path.join(ROOT_DIR, "dataset", "raw", "OD_dong_list.xlsx")
OUTPUT_DIR = os.path.join(GRAVITY_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAP_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gravity_prediction_map.html")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "지도_상위OD흐름.csv")
TOTALS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "동탄,위례,검단_총량비교.csv")
ALL_TEST_OD_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "동탄,위례,검단_OD비교.csv")


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    return numerator / denominator if denominator > 0 else 0.0


def save_csv_safely(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        fallback_path = f"{base}_new{ext}"
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"file is open, saved instead: {fallback_path}")
        return fallback_path


def load_centroids():
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    centroids = {}
    test_features = []
    for feature in geojson["features"]:
        props = feature["properties"]
        code = int(props["adm_cd8"])
        geom = shape(feature["geometry"])
        point = geom.representative_point()
        centroids[code] = {
            "lat": point.y,
            "lon": point.x,
            "name": props.get("adm_nm", str(code)),
        }
        test_features.append(feature)

    return geojson, centroids


def prepare_prediction():
    sys.path.insert(0, SRC_DIR)
    dataset_module = load_module("gravity_dataset", os.path.join(GRAVITY_DIR, "dataset.py"))
    model_module = load_module("gravity_model", os.path.join(GRAVITY_DIR, "model.py"))

    dataset = dataset_module.ODDataset()

    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    train_mask[dataset.test_indices] = False

    x_od = dataset.X_OD.copy()
    x_od[:, ~train_mask] = 0
    x_od[~train_mask, :] = 0

    y_o = np.sum(x_od, axis=1)
    y_d = np.sum(x_od, axis=0)
    y_self = np.diag(x_od)
    y_external_out = y_o - y_self
    y_external_in = y_d - y_self

    model = model_module.DoublyConstrainedGravityModel(beta=2.0, max_iter=100)
    t_pred = model.fit_predict(
        dataset.X_static[train_mask],
        y_external_out[train_mask],
        y_external_in[train_mask],
        dataset.X_static,
        dataset.X_dist,
        y_self[train_mask],
        None,
    )

    y_od_all = dataset.X_OD.copy()
    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, dataset.test_indices] = True
    test_mask_2d[dataset.test_indices, :] = True

    metrics = {
        "rmse_all": float(np.sqrt(np.mean((y_od_all - t_pred) ** 2))),
        "cpc_all": float(cpc_score(y_od_all, t_pred)),
        "rmse_test": float(np.sqrt(np.mean((y_od_all[test_mask_2d] - t_pred[test_mask_2d]) ** 2))),
        "cpc_test": float(cpc_score(y_od_all[test_mask_2d], t_pred[test_mask_2d])),
    }

    dong_df = pd.read_excel(DONG_LIST_PATH)
    dong_codes = dong_df["dong_code"].astype(int).to_numpy()
    dong_names = dong_df["dong_name"].astype(str).to_numpy()

    return dataset, t_pred, y_od_all, dong_codes, dong_names, metrics


def collect_top_flows(dataset, t_pred, actual_od, dong_codes, dong_names, centroids, top_n=80):
    rows = []
    test_set = set(dataset.test_indices.tolist())

    for i in range(dataset.num_nodes):
        for j in range(dataset.num_nodes):
            if i == j:
                continue
            if i not in test_set and j not in test_set:
                continue

            o_code = int(dong_codes[i])
            d_code = int(dong_codes[j])
            if o_code not in centroids or d_code not in centroids:
                continue

            pred = float(t_pred[i, j])
            actual = float(actual_od[i, j])
            if pred <= 0 and actual <= 0:
                continue

            if i in test_set and j in test_set:
                direction = "테스트구역 내부/상호"
            elif i in test_set:
                direction = "테스트구역에서 나감"
            else:
                direction = "테스트구역으로 들어옴"

            rows.append(
                {
                    "origin_code": o_code,
                    "origin_name": dong_names[i],
                    "dest_code": d_code,
                    "dest_name": dong_names[j],
                    "direction": direction,
                    "predicted": pred,
                    "actual": actual,
                    "abs_error": abs(pred - actual),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("predicted", ascending=False).head(top_n).reset_index(drop=True)
    save_csv_safely(df.rename(
        columns={
            "origin_code": "출발행정동코드",
            "origin_name": "출발행정동",
            "dest_code": "도착행정동코드",
            "dest_name": "도착행정동",
            "direction": "구분",
            "predicted": "예측통행량",
            "actual": "실제통행량",
            "abs_error": "절대오차",
        }
    ), CSV_OUTPUT_PATH)
    return df


def collect_all_test_od(dataset, t_pred, actual_od, dong_codes, dong_names):
    rows = []
    test_set = set(dataset.test_indices.tolist())

    for i in range(dataset.num_nodes):
        for j in range(dataset.num_nodes):
            if i not in test_set and j not in test_set:
                continue

            if i == j:
                direction = "내부통행"
            elif i in test_set and j in test_set:
                direction = "테스트구역 내부/상호"
            elif i in test_set:
                direction = "테스트구역에서 나감"
            else:
                direction = "테스트구역으로 들어옴"

            actual = float(actual_od[i, j])
            predicted = float(t_pred[i, j])
            rows.append(
                {
                    "출발행정동코드": int(dong_codes[i]),
                    "출발행정동": dong_names[i],
                    "도착행정동코드": int(dong_codes[j]),
                    "도착행정동": dong_names[j],
                    "구분": direction,
                    "예측통행량": predicted,
                    "실제통행량": actual,
                    "오차_예측-실제": predicted - actual,
                    "절대오차": abs(predicted - actual),
                    "상대오차": abs(predicted - actual) / max(actual, 1.0),
                    "거리": float(dataset.X_dist[i, j]),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("절대오차", ascending=False).reset_index(drop=True)
    save_csv_safely(df, ALL_TEST_OD_OUTPUT_PATH)
    return df


def collect_test_totals(dataset, t_pred, actual_od, dong_codes, dong_names):
    rows = []
    for idx in dataset.test_indices:
        actual_self = float(actual_od[idx, idx])
        pred_self = float(t_pred[idx, idx])
        actual_out = float(actual_od[idx].sum() - actual_self)
        pred_out = float(t_pred[idx].sum() - pred_self)
        actual_in = float(actual_od[:, idx].sum() - actual_self)
        pred_in = float(t_pred[:, idx].sum() - pred_self)

        rows.append(
            {
                "dong_code": int(dong_codes[idx]),
                "dong_name": dong_names[idx],
                "actual_external_out": actual_out,
                "pred_external_out": pred_out,
                "out_error": pred_out - actual_out,
                "out_abs_error": abs(pred_out - actual_out),
                "actual_external_in": actual_in,
                "pred_external_in": pred_in,
                "in_error": pred_in - actual_in,
                "in_abs_error": abs(pred_in - actual_in),
                "actual_self": actual_self,
                "pred_self": pred_self,
                "self_error": pred_self - actual_self,
                "self_abs_error": abs(pred_self - actual_self),
            }
        )

    df = pd.DataFrame(rows)
    save_csv_safely(df.rename(
        columns={
            "dong_code": "행정동코드",
            "dong_name": "행정동",
            "actual_external_out": "실제외부유출",
            "pred_external_out": "예측외부유출",
            "out_error": "외부유출오차",
            "out_abs_error": "외부유출절대오차",
            "actual_external_in": "실제외부유입",
            "pred_external_in": "예측외부유입",
            "in_error": "외부유입오차",
            "in_abs_error": "외부유입절대오차",
            "actual_self": "실제내부통행",
            "pred_self": "예측내부통행",
            "self_error": "내부통행오차",
            "self_abs_error": "내부통행절대오차",
        }
    ), TOTALS_OUTPUT_PATH)
    return df


def make_map(flow_df, totals_df, dataset, dong_codes, centroids, geojson, metrics):
    m = folium.Map(location=[37.45, 126.95], zoom_start=10, tiles="cartodbpositron")

    title_html = f"""
    <div style="position: fixed; top: 16px; left: 50px; z-index: 9999;
                background: white; border: 1px solid #aaa; border-radius: 6px;
                padding: 10px 12px; font-family: Malgun Gothic, Arial; font-size: 14px;">
      <b>중력모델 예측 OD 지도</b><br>
      beta = 2.0<br>
      전체 CPC: {metrics['cpc_all']:.4f}, 전체 RMSE: {metrics['rmse_all']:.1f}<br>
      Test CPC: {metrics['cpc_test']:.4f}, Test RMSE: {metrics['rmse_test']:.1f}<br>
      선 굵기 = 예측 통행량 크기
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    test_codes = {int(dong_codes[idx]) for idx in dataset.test_indices}

    test_geojson = {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in geojson["features"]
            if int(feature["properties"]["adm_cd8"]) in test_codes
        ],
    }

    folium.GeoJson(
        test_geojson,
        name="동탄/위례/검단 테스트 구역",
        style_function=lambda _: {
            "fillColor": "#f3c969",
            "color": "#7a4f00",
            "weight": 2,
            "fillOpacity": 0.45,
        },
        tooltip=folium.GeoJsonTooltip(fields=["adm_nm"], aliases=["테스트 행정동"]),
    ).add_to(m)

    outgoing = folium.FeatureGroup(name="테스트 구역에서 나가는 예측 OD", show=True)
    incoming = folium.FeatureGroup(name="테스트 구역으로 들어오는 예측 OD", show=True)
    internal = folium.FeatureGroup(name="테스트 구역 내부/상호 예측 OD", show=True)
    totals_layer = folium.FeatureGroup(name="테스트 동별 총량: 실제 vs 예측", show=True)

    max_pred = max(float(flow_df["predicted"].max()), 1.0)
    color_map = {
        "테스트구역에서 나감": "#d94841",
        "테스트구역으로 들어옴": "#2f6fbb",
        "테스트구역 내부/상호": "#7b3fb2",
    }
    group_map = {
        "테스트구역에서 나감": outgoing,
        "테스트구역으로 들어옴": incoming,
        "테스트구역 내부/상호": internal,
    }

    for _, row in flow_df.iterrows():
        o = centroids[int(row["origin_code"])]
        d = centroids[int(row["dest_code"])]
        weight = 1.2 + 8.0 * (float(row["predicted"]) / max_pred) ** 0.5
        color = color_map[row["direction"]]
        popup = f"""
        <div style="font-family: Malgun Gothic, Arial; font-size: 13px;">
          <b>{row['origin_name']} → {row['dest_name']}</b><br>
          구분: {row['direction']}<br>
          예측: {row['predicted']:.1f}<br>
          실제: {row['actual']:.1f}<br>
          절대오차: {row['abs_error']:.1f}
        </div>
        """
        folium.PolyLine(
            locations=[(o["lat"], o["lon"]), (d["lat"], d["lon"])],
            color=color,
            weight=weight,
            opacity=0.72,
            tooltip=f"{row['origin_name']} → {row['dest_name']}: 예측 {row['predicted']:.0f}",
            popup=folium.Popup(popup, max_width=360),
        ).add_to(group_map[row["direction"]])

    max_out = max(float(totals_df["pred_external_out"].max()), 1.0)
    for _, row in totals_df.iterrows():
        code = int(row["dong_code"])
        if code not in centroids:
            continue

        c = centroids[code]
        radius = 6 + 18 * (float(row["pred_external_out"]) / max_out) ** 0.5
        out_error_rate = row["out_error"] / row["actual_external_out"] if row["actual_external_out"] > 0 else 0
        in_error_rate = row["in_error"] / row["actual_external_in"] if row["actual_external_in"] > 0 else 0
        self_error_rate = row["self_error"] / row["actual_self"] if row["actual_self"] > 0 else 0

        popup = f"""
        <div style="font-family: Malgun Gothic, Arial; font-size: 13px; min-width: 280px;">
          <b>{row['dong_name']} ({code})</b><br><br>
          <b>외부 유출 총합</b><br>
          실제: {row['actual_external_out']:.1f}<br>
          예측: {row['pred_external_out']:.1f}<br>
          오차: {row['out_error']:+.1f} ({out_error_rate:+.1%})<br><br>
          <b>외부 유입 총합</b><br>
          실제: {row['actual_external_in']:.1f}<br>
          예측: {row['pred_external_in']:.1f}<br>
          오차: {row['in_error']:+.1f} ({in_error_rate:+.1%})<br><br>
          <b>내부통행</b><br>
          실제: {row['actual_self']:.1f}<br>
          예측: {row['pred_self']:.1f}<br>
          오차: {row['self_error']:+.1f} ({self_error_rate:+.1%})
        </div>
        """
        tooltip = (
            f"{row['dong_name']} | 외부유출 실제 {row['actual_external_out']:.0f}, "
            f"예측 {row['pred_external_out']:.0f}"
        )
        folium.CircleMarker(
            location=(c["lat"], c["lon"]),
            radius=radius,
            color="#111111",
            weight=1.5,
            fill=True,
            fill_color="#ffd166",
            fill_opacity=0.82,
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=420),
        ).add_to(totals_layer)

    outgoing.add_to(m)
    incoming.add_to(m)
    internal.add_to(m)
    totals_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(MAP_OUTPUT_PATH)


def main():
    geojson, centroids = load_centroids()
    dataset, t_pred, actual_od, dong_codes, dong_names, metrics = prepare_prediction()
    flow_df = collect_top_flows(dataset, t_pred, actual_od, dong_codes, dong_names, centroids)
    collect_all_test_od(dataset, t_pred, actual_od, dong_codes, dong_names)
    totals_df = collect_test_totals(dataset, t_pred, actual_od, dong_codes, dong_names)
    make_map(flow_df, totals_df, dataset, dong_codes, centroids, geojson, metrics)
    print(f"saved map: {MAP_OUTPUT_PATH}")
    print(f"saved top flows: {CSV_OUTPUT_PATH}")
    print(f"saved all test od: {ALL_TEST_OD_OUTPUT_PATH}")
    print(f"saved test totals: {TOTALS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
