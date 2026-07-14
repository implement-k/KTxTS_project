import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_OK"] = "True"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DONG_CODE_PATH
from dataset import ODDataset
from model import DoublyConstrainedGravityModel


출력폴더 = Path(__file__).resolve().parent / "outputs"
전체_OD_파일 = 출력폴더 / "전체_1137개_행정동_OD_중력모델_예측.csv"
테스트_OD_파일 = 출력폴더 / "테스트지역포함_OD_중력모델_예측.csv"
전체_총량_파일 = 출력폴더 / "전체_행정동_총량_중력모델_예측.csv"
테스트_총량_파일 = 출력폴더 / "테스트지역_총량_중력모델_예측.csv"


테스트지역코드 = {
    "동탄": [
        31240600,
        31240610,
        31240620,
        31240640,
        31240650,
        31240690,
        31240700,
        31240710,
    ],
    "위례": [
        11240820,
        31021680,
        31180650,
    ],
    "검단": [
        23080800,
        23080810,
        23080850,
        23080860,
        23080870,
        23080880,
    ],
}


def 지역명_찾기(행정동코드: int) -> str:
    for 지역명, 코드목록 in 테스트지역코드.items():
        if int(행정동코드) in 코드목록:
            return 지역명
    return ""


def cpc_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    분자 = 2 * np.sum(np.minimum(y_true, y_pred))
    분모 = np.sum(y_true) + np.sum(y_pred)
    return 분자 / 분모 if 분모 > 0 else 0.0


def 모델_학습_및_예측():
    데이터셋 = ODDataset()

    학습마스크 = np.ones(데이터셋.num_nodes, dtype=bool)
    학습마스크[데이터셋.test_indices] = False

    학습용_OD = 데이터셋.X_OD.copy()
    학습용_OD[:, ~학습마스크] = 0
    학습용_OD[~학습마스크, :] = 0

    내부통행_학습 = np.diag(학습용_OD)
    외부유출_학습 = np.sum(학습용_OD, axis=1) - 내부통행_학습
    외부유입_학습 = np.sum(학습용_OD, axis=0) - 내부통행_학습

    보정계수 = 데이터셋.num_nodes / len(데이터셋.train_indices)
    X_static_train = 데이터셋.X_static[학습마스크]
    O_train = 외부유출_학습[학습마스크] * 보정계수
    D_train = 외부유입_학습[학습마스크] * 보정계수
    self_train = 내부통행_학습[학습마스크] * 보정계수
    inter_train = 외부유출_학습[학습마스크] * 보정계수

    모델 = DoublyConstrainedGravityModel(beta=1.5, max_iter=100)
    모델.fit_lgbm_O_D(X_static_train, O_train, D_train)
    모델.fit_lgbm_self_inter(X_static_train, self_train, inter_train)

    예측_외부유출, 예측_외부유입 = 모델.predict_O_D(데이터셋.X_static)
    예측_내부통행, 예측_inter = 모델.predict_self_inter(데이터셋.X_static)
    예측_OD = 모델.apply_ipf(
        예측_외부유출,
        예측_외부유입,
        데이터셋.X_dist,
        y_self=예측_내부통행,
        y_inter=예측_inter,
    )

    return 데이터셋, 학습마스크, 예측_OD, 예측_외부유출, 예측_외부유입, 예측_내부통행


def 행정동정보_불러오기() -> pd.DataFrame:
    행정동 = pd.read_excel(DONG_CODE_PATH)
    행정동 = 행정동[["dong_code", "dong_name"]].copy()
    행정동["dong_code"] = 행정동["dong_code"].astype(int)
    행정동["테스트지역명"] = 행정동["dong_code"].map(지역명_찾기)
    행정동["테스트지역여부"] = 행정동["테스트지역명"].ne("")
    return 행정동


def OD_결과표_만들기(데이터셋, 학습마스크, 예측_OD, 행정동: pd.DataFrame) -> pd.DataFrame:
    행정동코드 = 행정동["dong_code"].to_numpy(dtype=np.int64)
    행정동명 = 행정동["dong_name"].astype(str).to_numpy()
    테스트지역명 = 행정동["테스트지역명"].astype(str).to_numpy()
    테스트여부 = 행정동["테스트지역여부"].to_numpy(dtype=bool)

    출발_idx, 도착_idx = np.meshgrid(
        np.arange(데이터셋.num_nodes),
        np.arange(데이터셋.num_nodes),
        indexing="ij",
    )

    실제 = 데이터셋.X_OD.reshape(-1).astype(np.float64)
    예측 = 예측_OD.reshape(-1).astype(np.float64)
    출발_flat = 출발_idx.reshape(-1)
    도착_flat = 도착_idx.reshape(-1)

    결과 = pd.DataFrame(
        {
            "출발행정동코드": 행정동코드[출발_flat],
            "출발행정동": 행정동명[출발_flat],
            "도착행정동코드": 행정동코드[도착_flat],
            "도착행정동": 행정동명[도착_flat],
            "실제통행량": np.round(실제, 3),
            "예측통행량_중력모델": np.round(예측, 3),
            "오차_예측-실제": np.round(예측 - 실제, 3),
            "절대오차": np.round(np.abs(예측 - 실제), 3),
            "출발테스트지역여부": 테스트여부[출발_flat],
            "도착테스트지역여부": 테스트여부[도착_flat],
            "테스트지역포함여부": 테스트여부[출발_flat] | 테스트여부[도착_flat],
            "출발테스트지역명": 테스트지역명[출발_flat],
            "도착테스트지역명": 테스트지역명[도착_flat],
            "학습구분": np.where(
                테스트여부[출발_flat] | 테스트여부[도착_flat],
                "테스트지역포함",
                "학습지역간",
            ),
        }
    )
    return 결과


def 총량_결과표_만들기(
    데이터셋,
    예측_OD,
    행정동: pd.DataFrame,
    예측_외부유출_LGBM,
    예측_외부유입_LGBM,
    예측_내부통행_LGBM,
) -> pd.DataFrame:
    실제_내부 = np.diag(데이터셋.X_OD).astype(np.float64)
    예측_내부 = np.diag(예측_OD).astype(np.float64)
    예측_외부유출_LGBM = np.asarray(예측_외부유출_LGBM, dtype=np.float64)
    예측_외부유입_LGBM = np.asarray(예측_외부유입_LGBM, dtype=np.float64)
    예측_내부통행_LGBM = np.asarray(예측_내부통행_LGBM, dtype=np.float64)

    실제_전체유출 = np.sum(데이터셋.X_OD, axis=1).astype(np.float64)
    실제_전체유입 = np.sum(데이터셋.X_OD, axis=0).astype(np.float64)
    예측_전체유출 = np.sum(예측_OD, axis=1).astype(np.float64)
    예측_전체유입 = np.sum(예측_OD, axis=0).astype(np.float64)

    실제_외부유출 = 실제_전체유출 - 실제_내부
    실제_외부유입 = 실제_전체유입 - 실제_내부
    예측_외부유출 = 예측_전체유출 - 예측_내부
    예측_외부유입 = 예측_전체유입 - 예측_내부

    결과 = 행정동.rename(
        columns={"dong_code": "행정동코드", "dong_name": "행정동"}
    ).copy()
    결과["실제전체유출"] = 실제_전체유출
    결과["예측전체유출_최종OD"] = 예측_전체유출
    결과["전체유출오차_예측-실제"] = 예측_전체유출 - 실제_전체유출
    결과["실제외부유출"] = 실제_외부유출
    결과["예측외부유출_LGBM"] = 예측_외부유출_LGBM
    결과["예측외부유출_최종OD행합"] = 예측_외부유출
    결과["외부유출오차_예측-실제"] = 예측_외부유출 - 실제_외부유출
    결과["실제전체유입"] = 실제_전체유입
    결과["예측전체유입_최종OD"] = 예측_전체유입
    결과["전체유입오차_예측-실제"] = 예측_전체유입 - 실제_전체유입
    결과["실제외부유입"] = 실제_외부유입
    결과["예측외부유입_LGBM"] = 예측_외부유입_LGBM
    결과["예측외부유입_최종OD열합"] = 예측_외부유입
    결과["외부유입오차_예측-실제"] = 예측_외부유입 - 실제_외부유입
    결과["실제내부통행"] = 실제_내부
    결과["예측내부통행_LGBM"] = 예측_내부통행_LGBM
    결과["예측내부통행_최종OD대각"] = 예측_내부
    결과["내부통행오차_예측-실제"] = 예측_내부 - 실제_내부

    수치컬럼 = 결과.select_dtypes(include=[np.number]).columns.difference(["행정동코드"])
    결과[수치컬럼] = 결과[수치컬럼].round(3)
    return 결과


def 저장하기():
    출력폴더.mkdir(parents=True, exist_ok=True)

    데이터셋, 학습마스크, 예측_OD, 예측_외부유출, 예측_외부유입, 예측_내부통행 = 모델_학습_및_예측()
    행정동 = 행정동정보_불러오기()

    전체_OD = OD_결과표_만들기(데이터셋, 학습마스크, 예측_OD, 행정동)
    테스트_OD = 전체_OD[전체_OD["테스트지역포함여부"]].copy()

    전체_총량 = 총량_결과표_만들기(
        데이터셋,
        예측_OD,
        행정동,
        예측_외부유출,
        예측_외부유입,
        예측_내부통행,
    )
    테스트_총량 = 전체_총량[전체_총량["테스트지역여부"]].copy()

    전체_OD.to_csv(전체_OD_파일, index=False, encoding="utf-8-sig")
    테스트_OD.to_csv(테스트_OD_파일, index=False, encoding="utf-8-sig")
    전체_총량.to_csv(전체_총량_파일, index=False, encoding="utf-8-sig")
    테스트_총량.to_csv(테스트_총량_파일, index=False, encoding="utf-8-sig")

    테스트_RMSE = np.sqrt(
        np.mean(
            (
                테스트_OD["실제통행량"].to_numpy(dtype=np.float64)
                - 테스트_OD["예측통행량_중력모델"].to_numpy(dtype=np.float64)
            )
            ** 2
        )
    )
    테스트_CPC = cpc_score(
        테스트_OD["실제통행량"].to_numpy(dtype=np.float64),
        테스트_OD["예측통행량_중력모델"].to_numpy(dtype=np.float64),
    )

    print(f"전체 OD 저장: {전체_OD_파일} ({len(전체_OD):,}행)")
    print(f"테스트지역 포함 OD 저장: {테스트_OD_파일} ({len(테스트_OD):,}행)")
    print(f"전체 총량 저장: {전체_총량_파일} ({len(전체_총량):,}행)")
    print(f"테스트지역 총량 저장: {테스트_총량_파일} ({len(테스트_총량):,}행)")
    print(f"테스트지역 포함 OD RMSE: {테스트_RMSE:.3f}")
    print(f"테스트지역 포함 OD CPC: {테스트_CPC:.4f}")


if __name__ == "__main__":
    저장하기()
