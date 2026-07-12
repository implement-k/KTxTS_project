import importlib.util
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd


GRAVITY_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(GRAVITY_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
OUTPUT_DIR = GRAVITY_DIR
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "beta_tuning_results.csv")
FIG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "beta_tuning_results.png")
DIST_CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "beta_distance_bin_results.csv")
DIST_FIG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "beta_distance_bin_results.png")
BETA_VALUES = [1.0, 1.5, 2.0, 2.25, 2.5]
DISTANCE_BINS = [
    ("0~5km", 0, 5),
    ("5~10km", 5, 10),
    ("10~20km", 10, 20),
    ("20~40km", 20, 40),
    ("40km+", 40, float("inf")),
]


def setup_korean_font():
    font_path = r"C:\Windows\Fonts\malgun.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rcParams["font.family"] = font_name
    rcParams["axes.unicode_minus"] = False


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


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def prepare_data():
    sys.path.insert(0, SRC_DIR)
    dataset_module = load_module("gravity_dataset_beta_tuning", os.path.join(GRAVITY_DIR, "dataset.py"))
    model_module = load_module("gravity_model_beta_tuning", os.path.join(GRAVITY_DIR, "model.py"))

    dataset = dataset_module.ODDataset()
    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    train_mask[dataset.test_indices] = False

    x_od = dataset.X_OD.copy()
    x_od[:, ~train_mask] = 0
    x_od[~train_mask, :] = 0

    y_o = x_od.sum(axis=1)
    y_d = x_od.sum(axis=0)
    y_self = np.diag(x_od)
    y_external_out = y_o - y_self
    y_external_in = y_d - y_self

    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, dataset.test_indices] = True
    test_mask_2d[dataset.test_indices, :] = True
    test_external_mask_2d = test_mask_2d.copy()
    np.fill_diagonal(test_external_mask_2d, False)

    return {
        "dataset": dataset,
        "model_module": model_module,
        "train_mask": train_mask,
        "X_static_train": dataset.X_static[train_mask],
        "O_train": y_external_out[train_mask],
        "D_train": y_external_in[train_mask],
        "X_self": y_self[train_mask],
        "actual_all": dataset.X_OD.copy().astype(float),
        "test_mask_2d": test_mask_2d,
        "test_external_mask_2d": test_external_mask_2d,
    }


def evaluate_beta(beta, data):
    dataset = data["dataset"]
    model_module = data["model_module"]

    model = model_module.DoublyConstrainedGravityModel(beta=beta, max_iter=100)
    t_pred = model.fit_predict(
        data["X_static_train"],
        data["O_train"],
        data["D_train"],
        dataset.X_static,
        dataset.X_dist,
        data["X_self"],
        None,
    ).astype(float)

    actual_all = data["actual_all"]
    test_mask = data["test_mask_2d"]
    actual_test = actual_all[test_mask]
    pred_test = t_pred[test_mask]
    mask_1000 = actual_test > 1000

    summary = {
        "beta": beta,
        "all_rmse": rmse(actual_all, t_pred),
        "all_cpc": float(cpc_score(actual_all, t_pred)),
        "test_rmse": rmse(actual_test, pred_test),
        "test_cpc": float(cpc_score(actual_test, pred_test)),
        "over_1000_rmse": rmse(actual_test[mask_1000], pred_test[mask_1000]),
        "over_1000_cpc": float(cpc_score(actual_test[mask_1000], pred_test[mask_1000])),
        "over_1000_n": int(mask_1000.sum()),
    }

    distance_rows = []
    external_mask = data["test_external_mask_2d"]
    actual_ext = actual_all[external_mask]
    pred_ext = t_pred[external_mask]
    dist_ext = dataset.X_dist[external_mask].astype(float)

    for label, lo, hi in DISTANCE_BINS:
        if np.isfinite(hi):
            dist_mask = (dist_ext > lo) & (dist_ext <= hi)
        else:
            dist_mask = dist_ext > lo

        if not np.any(dist_mask):
            distance_rows.append(
                {
                    "beta": beta,
                    "distance_bin": label,
                    "n": 0,
                    "true_sum": 0.0,
                    "pred_sum": 0.0,
                    "pred_true_ratio": 0.0,
                    "rmse": 0.0,
                    "cpc": 0.0,
                }
            )
            continue

        y_true_bin = actual_ext[dist_mask]
        y_pred_bin = pred_ext[dist_mask]
        true_sum = float(y_true_bin.sum())
        pred_sum = float(y_pred_bin.sum())

        distance_rows.append(
            {
                "beta": beta,
                "distance_bin": label,
                "n": int(dist_mask.sum()),
                "true_sum": true_sum,
                "pred_sum": pred_sum,
                "pred_true_ratio": pred_sum / true_sum if true_sum > 0 else 0.0,
                "rmse": rmse(y_true_bin, y_pred_bin),
                "cpc": float(cpc_score(y_true_bin, y_pred_bin)),
            }
        )

    return summary, distance_rows


def plot_results(df):
    setup_korean_font()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    x = df["beta"].astype(str)
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]

    axes[0, 0].bar(x, df["test_cpc"], color=colors)
    axes[0, 0].set_title("Test CPC")
    axes[0, 0].set_xlabel("beta")
    axes[0, 0].set_ylabel("높을수록 좋음")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=0.25)
    for i, value in enumerate(df["test_cpc"]):
        axes[0, 0].text(i, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    axes[0, 1].bar(x, df["test_rmse"], color=colors)
    axes[0, 1].set_title("Test RMSE")
    axes[0, 1].set_xlabel("beta")
    axes[0, 1].set_ylabel("낮을수록 좋음")
    axes[0, 1].grid(alpha=0.25)
    for i, value in enumerate(df["test_rmse"]):
        axes[0, 1].text(i, value + 4, f"{value:.1f}", ha="center", fontsize=9)

    axes[1, 0].bar(x, df["over_1000_cpc"], color=colors)
    axes[1, 0].set_title("1000명 초과 OD CPC")
    axes[1, 0].set_xlabel("beta")
    axes[1, 0].set_ylabel("높을수록 좋음")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(axis="y", alpha=0.25)
    for i, value in enumerate(df["over_1000_cpc"]):
        axes[1, 0].text(i, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    axes[1, 1].axis("off")
    table_df = df[
        ["beta", "all_rmse", "all_cpc", "test_rmse", "test_cpc", "over_1000_cpc"]
    ].copy()
    table_df.columns = ["beta", "전체 RMSE", "전체 CPC", "Test RMSE", "Test CPC", "1000+ CPC"]
    for col in ["전체 RMSE", "Test RMSE"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.1f}")
    for col in ["전체 CPC", "Test CPC", "1000+ CPC"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.3f}")
    table_df["beta"] = table_df["beta"].map(lambda v: f"{v:g}")
    table = axes[1, 1].table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.55)
    axes[1, 1].set_title("요약표", pad=12)

    fig.suptitle("beta 튜닝 결과: 동탄 + 위례 + 검단 테스트 구역", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_OUTPUT_PATH, bbox_inches="tight")


def plot_distance_results(dist_df):
    setup_korean_font()
    bins = [item[0] for item in DISTANCE_BINS]
    betas = BETA_VALUES

    ratio_matrix = (
        dist_df.pivot(index="beta", columns="distance_bin", values="pred_true_ratio")
        .loc[betas, bins]
        .to_numpy()
    )
    cpc_matrix = (
        dist_df.pivot(index="beta", columns="distance_bin", values="cpc")
        .loc[betas, bins]
        .to_numpy()
    )
    rmse_matrix = (
        dist_df.pivot(index="beta", columns="distance_bin", values="rmse")
        .loc[betas, bins]
        .to_numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=140)

    def draw_heatmap(ax, matrix, title, cmap, fmt, vmin=None, vmax=None):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(bins)), labels=bins)
        ax.set_yticks(np.arange(len(betas)), labels=[str(beta) for beta in betas])
        ax.set_xlabel("거리 구간")
        ax.set_ylabel("beta")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    draw_heatmap(
        axes[0],
        ratio_matrix,
        "예측총량 / 실제총량\n1에 가까울수록 좋음",
        "RdYlGn_r",
        "{:.2f}",
        vmin=0.4,
        vmax=1.8,
    )
    draw_heatmap(
        axes[1],
        cpc_matrix,
        "거리 구간별 CPC\n높을수록 좋음",
        "Greens",
        "{:.2f}",
        vmin=0,
        vmax=1,
    )
    draw_heatmap(
        axes[2],
        rmse_matrix,
        "거리 구간별 RMSE\n낮을수록 좋음",
        "Reds",
        "{:.0f}",
    )

    fig.suptitle("beta별 거리 구간 성능: 테스트 구역 외부 OD", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(DIST_FIG_OUTPUT_PATH, bbox_inches="tight")


def main():
    data = prepare_data()
    rows = []
    distance_rows = []
    for beta in BETA_VALUES:
        print(f"\n=== beta={beta} ===")
        row, beta_distance_rows = evaluate_beta(beta, data)
        rows.append(row)
        distance_rows.extend(beta_distance_rows)
        print(
            f"All RMSE={row['all_rmse']:.4f}, All CPC={row['all_cpc']:.4f}, "
            f"Test RMSE={row['test_rmse']:.4f}, Test CPC={row['test_cpc']:.4f}, "
            f"1000+ RMSE={row['over_1000_rmse']:.4f}, "
            f"1000+ CPC={row['over_1000_cpc']:.4f}"
        )

    df = pd.DataFrame(rows)
    dist_df = pd.DataFrame(distance_rows)
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    dist_df.to_csv(DIST_CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    plot_results(df)
    plot_distance_results(dist_df)

    print("\n=== beta tuning summary ===")
    print(df.to_string(index=False))
    print(f"\nsaved csv: {CSV_OUTPUT_PATH}")
    print(f"saved figure: {FIG_OUTPUT_PATH}")
    print(f"saved distance csv: {DIST_CSV_OUTPUT_PATH}")
    print(f"saved distance figure: {DIST_FIG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
