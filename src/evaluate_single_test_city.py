import os
import sys

os.environ["KMP_DUPLICATE_OK"] = "True"

import numpy as np
import pandas as pd

GRAVITY_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(GRAVITY_DIR)
sys.path.insert(0, SRC_DIR)

from config import DONG_CODE_PATH
from dataset import ODDataset
from model import DoublyConstrainedGravityModel


OUTPUT_DIR = os.path.join(GRAVITY_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "동탄,위례,검단_단일마스킹_평가결과.csv")

SINGLE_TEST_CITIES = {
    "동탄": [
        "31240600",  # 동탄2동
        "31240610",  # 동탄1동
        "31240620",  # 동탄3동
        "31240640",  # 동탄4동
        "31240650",  # 동탄5동
        "31240690",  # 동탄7동
        "31240700",  # 동탄6동
        "31240710",  # 동탄8동
    ],
    "위례": [
        "11240820",
        "31021680",
        "31180650",
    ],
    "검단": [
        "23080800",  # 검단동
        "23080810",  # 불로대곡동
        "23080850",  # 당하동
        "23080860",  # 마전동
        "23080870",  # 원당동
        "23080880",  # 아라동
    ],
}


def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    return numerator / denominator if denominator > 0 else 0.0


def get_city_indices(city_codes):
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dong_codes = dong_df["dong_code"].astype(str).to_numpy()
    code_to_idx = {code: idx for idx, code in enumerate(dong_codes)}
    indices = [code_to_idx[code] for code in city_codes if code in code_to_idx]
    return np.array(indices, dtype=int)


def evaluate_one_city(dataset, city_name, city_indices):
    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    train_mask[city_indices] = False

    x_od = dataset.X_OD.copy()
    x_od[:, ~train_mask] = 0
    x_od[~train_mask, :] = 0

    y_o = np.sum(x_od, axis=1)
    y_d = np.sum(x_od, axis=0)
    y_self = np.diag(x_od)
    y_external_out = y_o - y_self
    y_external_in = y_d - y_self

    model = DoublyConstrainedGravityModel(beta=2.0, max_iter=100)
    t_pred = model.fit_predict(
        dataset.X_static[train_mask],
        y_external_out[train_mask],
        y_external_in[train_mask],
        dataset.X_static,
        dataset.X_dist,
        y_self[train_mask],
        None,
    )

    actual_od = dataset.X_OD.copy()

    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, city_indices] = True
    test_mask_2d[city_indices, :] = True

    diagonal_mask = np.eye(dataset.num_nodes, dtype=bool)
    test_external_mask = test_mask_2d & ~diagonal_mask
    test_self_mask = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_self_mask[city_indices, city_indices] = True

    return {
        "가린지역": city_name,
        "테스트행정동수": int(len(city_indices)),
        "전체_RMSE": float(np.sqrt(np.mean((actual_od - t_pred) ** 2))),
        "전체_CPC": float(cpc_score(actual_od, t_pred)),
        "Test_RMSE": float(np.sqrt(np.mean((actual_od[test_mask_2d] - t_pred[test_mask_2d]) ** 2))),
        "Test_CPC": float(cpc_score(actual_od[test_mask_2d], t_pred[test_mask_2d])),
        "Test외부_RMSE": float(
            np.sqrt(np.mean((actual_od[test_external_mask] - t_pred[test_external_mask]) ** 2))
        ),
        "Test외부_CPC": float(cpc_score(actual_od[test_external_mask], t_pred[test_external_mask])),
        "내부통행_RMSE": float(
            np.sqrt(np.mean((actual_od[test_self_mask] - t_pred[test_self_mask]) ** 2))
        ),
        "내부통행_CPC": float(cpc_score(actual_od[test_self_mask], t_pred[test_self_mask])),
        "실제Test총량": float(np.sum(actual_od[test_mask_2d])),
        "예측Test총량": float(np.sum(t_pred[test_mask_2d])),
    }


def main():
    dataset = ODDataset()
    rows = []

    for city_name, city_codes in SINGLE_TEST_CITIES.items():
        city_indices = get_city_indices(city_codes)
        if len(city_indices) == 0:
            raise ValueError(f"{city_name} 테스트 행정동 코드를 찾지 못했습니다.")
        print(f"\n=== {city_name}만 가리고 평가 ===")
        rows.append(evaluate_one_city(dataset, city_name, city_indices))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n" + result_df.to_string(index=False))
    print(f"\nsaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
