# MAE Django 백엔드 어댑터

## 백엔드 담당자용 1분 요약

| 항목 | 현재 상태 |
|---|---|
| 기준 브랜치 | `feature/mae-backend-adapter` |
| 기준 모델 | `feature/manage-version`의 `best_model/mae:v7.pth` |
| 모델 로딩 | 실제 체크포인트 strict loading 검증 완료 |
| 합성 입력 추론 | 실제 가중치와 LightGBM으로 검증 완료 |
| 실제 대상 신도시 추론 | 아직 불가 |
| 현재 node 수 | 1,137개 고정 |
| 백엔드 호출 인터페이스 | 구현 완료 |
| 프로젝트 전처리기 | 다음 연동 단계에서 필요 |
| 권장 실행 환경 | CPU |
| Linux 운영 경로 | 백엔드 환경 smoke test 필요 |

### 용어

- **node**: OD 행렬의 행과 열 하나에 대응하는 공간 단위다. 현재 `mae:v7.pth`에서는
  2021년 수도권 행정동 1개가 node 1개이며 총 1,137개다.
- **신도시 zone**: 신도시 계획 영역을 모델 입력용으로 나눈 공간 단위다. 공식 행정동과
  반드시 같지는 않으며 공공청사·도로·하천·능선·기존 행정동 경계 등을 고려해 구성할 수
  있다. zone 구성이 확정되면 각각 모델의 node로 사용할 수 있다.
- **N**: 한 시나리오의 모델 입력에 포함되는 전체 node 수다. 모델의 원본 출력은 이 N에
  대한 `N×N` OD 행렬이다.
- **가변-node 지원**: 신도시 zone 추가나 기존 공간 단위 조정으로 N이 달라져도 모델이
  처리할 수 있다는 뜻이다.

예를 들어 기존 1,137개 node를 모두 유지하면서 신도시 zone 4개를 추가하면 N은 1,141이고
원본 출력은 `1,141×1,141` OD 행렬이 된다. 다만 기존 행정동을 분리·대체·통합하면 최종
N은 단순히 `1,137+zone 수`와 다를 수 있다.

백엔드는 현재 브랜치로 모델 로딩과 호출 인터페이스를 연동할 수 있다. 다만 실제 대상 신도시
결과를 만들기 위한 zone·feature·scaler와 가변-node 지원 모델 코드·호환 체크포인트는
다음 연동 단계에서 전달 및 검증이 필요하다. 현재 버전은 백엔드 계약과 모델 실행 경로를
먼저 맞추는 기준 구현이다.
어댑터 인터페이스는 특정 신도시에 종속되지 않으며, 실제 지원 가능한 신도시는 향후
주입되는 전처리 설정과 모델 지원 범위에 따라 결정된다.

### 지금 각 담당자가 할 일

| 담당 | 지금 해야 할 일 |
|---|---|
| 백엔드 | 브랜치 checkout, 패키지 설치, Linux smoke test, Django singleton 연결 |
| AI | 확정 zone·node 순서·feature·scaler·연령/인구 정책과 가변-node 지원 코드·체크포인트 전달 |
| 어댑터 | 체크포인트 로딩, 모델 forward, OD 필터링, JSON 반환 |

## 빠른 시작

### 1. Git 브랜치로 받기

현재 기준 commit은 `d820686e9ad431ae2050a82c84600c8eeecf6b0e`이다. 이 commit에는
다음 모델 의존 파일이 이미 Git 추적 파일로 들어 있다.

- `best_model/mae:v7.pth`
- `best_model/best_lgbm_self_loop.txt`
- `src/mae/models.py`

`mae_backend_adapter/`는 아직 commit 전 untracked 상태다. 이 폴더를
`feature/mae-backend-adapter`의 후속 commit에 포함한 뒤 브랜치 전체를 전달하면 별도 모델
파일 복사는 필요 없다.

### 2. 패키지 설치

저장소 루트에서 실행한다.

```bash
python -m pip install -r mae_backend_adapter/requirements.txt
```

직접 필요한 runtime dependency는 `torch`, `lightgbm`, `numpy`뿐이다. pandas,
scikit-learn, matplotlib, SHAP 같은 데이터·학습 패키지는 사용하지 않는다.

### 3. 현재 검증된 합성 입력 smoke test

아래 코드는 실제 체크포인트와 LightGBM을 사용하지만 node code와 입력 tensor는 합성값이다.
실제 신도시 예제가 아니다.

```python
import torch

from mae_backend_adapter import MAEPredictor, ModelInputs

predictor = MAEPredictor(
    model_path="best_model/mae:v7.pth",
    device="cpu",
    supported_newtowns=("예시신도시",),
)
codes = tuple(f"SYNTHETIC_{index:04d}" for index in range(1137))
inputs = ModelInputs(
    x_static=torch.zeros(1137, 15),
    x_od_masked=torch.zeros(1137, 1137),
    x_dist=torch.zeros(1137, 1137),
    mask=torch.tensor([True] + [False] * 1136),
    origin_codes=codes,
    destination_codes=codes,
    newtown_zone_codes=("SYNTHETIC_0000",),
    population_allocation_method="synthetic_tensor_smoke_only",
    output_transform="log1p",
)
result = predictor.predict_from_tensors(
    inputs,
    request_metadata={"newtown": "예시신도시"},
)
```

### 4. Django 프로세스별 singleton 연결

predictor는 각 Gunicorn worker 프로세스 안에서 최초 요청 시 한 번 생성해 재사용한다.

```python
from functools import lru_cache

from mae_backend_adapter import MAEPredictor


@lru_cache(maxsize=1)
def get_mae_predictor():
    return MAEPredictor(
        model_path="best_model/mae:v7.pth",
        device="cpu",
        preprocessor=project_preprocessor,  # 다음 연동 단계에서 구현 후 주입
    )
```

### 5. 프로젝트 전처리기 준비 후 사용할 고수준 호출

`project_preprocessor`는 현재 제공되는 완성 객체가 아니다. 확정 zone, 전체 node 순서,
15개 feature, scaler, 인구·연령 정책을 구현한 뒤 주입해야 한다. 전처리기가 없거나 설정이
불완전하면 `PreprocessingConfigurationError`가 발생하는 것이 정상이다.

```python
result = get_mae_predictor().predict(
    newtown="예시신도시",
    total_population=100000,
    age_ratios={
        "0_19": 0.20,
        "20_39": 0.30,
        "40_64": 0.35,
        "65_plus": 0.15,
    },
)
```

## 백엔드 반환 형식

모델 내부에서는 전체 `N×N` OD를 검증하지만 기본 응답에는 origin 또는 destination이
신도시 zone인 항목만 포함한다. 행은 origin, 열은 destination이다.

```python
{
    "newtown": "예시신도시",
    "newtown_zone_codes": ["SAMPLE_NT_01", "SAMPLE_NT_02"],
    "od": [
        {
            "origin_code": "SAMPLE_NT_01",
            "destination_code": "OLD_CODE",
            "predicted_trips": 123.4,
            "movement_type": "outflow",
        },
        {
            "origin_code": "OLD_CODE",
            "destination_code": "SAMPLE_NT_01",
            "predicted_trips": 45.6,
            "movement_type": "inflow",
        },
        {
            "origin_code": "SAMPLE_NT_01",
            "destination_code": "SAMPLE_NT_02",
            "predicted_trips": 12.3,
            "movement_type": "internal",
        },
    ],
    "metadata": {
        "model_version": "mae:v7",
        "checkpoint": "mae:v7.pth",
        "node_count": 1137,
        "device": "cpu",
        "population_allocation_method": "official_allocation_version",
        "self_loop_policy": "lightgbm_override",
        "lightgbm_execution": "in_process",
        "negative_values_policy": "preserved_off_diagonal",
    },
}
```

- 신도시→신도시: `internal`
- 신도시→기존 지역: `outflow`
- 기존 지역→신도시: `inflow`

진단용 전체 행렬이 필요할 때만 `predict_full_matrix_from_tensors()`를 명시적으로 호출한다.

### 음수 OD 정책

현재 어댑터는 `expm1` 이후 LightGBM 대각값만 0 이상으로 처리하고 off-diagonal 음수는
보존하며 metadata에 `preserved_off_diagonal`을 기록한다. 반면 기존
`src/mae/model_test.py`는 지표 계산과 CSV 생성 시 전체 예측값을 0 이상으로 clamp한다.
실제 서비스에서 음수를 0으로 처리할지는 AI·백엔드 팀 합의가 필요하며, 합계 계산이나 지도
표시 전에 이 정책을 확정해야 한다.

## 현재 제한과 다음 연동 조건

**완료**

- Django용 호출 인터페이스
- `mae:v7.pth` strict loading
- 1,137-node 합성 입력의 실제 가중치 추론
- LightGBM 대각 처리
- 동일 predictor 반복 호출과 OD 필터·JSON 후처리

**현재 미완료**

- 신도시별 실제 입력 생성과 실제 신도시 OD 추론
- 신도시 zone 추가로 N이 달라지는 가변-node 추론

**다음 연동 단계에 필요한 자료**

- 대상 신도시별 확정 zone ID·경계·소속
- 신도시별 전체 node 순서와 static feature
- 신도시별 계획인구 배분과 연령 변환 정책
- 학습 당시 StandardScaler 통계 또는 정확히 재현할 동결 데이터와 분할
- API 연령 구간을 모델 연령 feature로 바꾸는 공식 정책
- 가변-node 지원 모델 코드와 호환 체크포인트의 전달 및 연동 검증
- 백엔드 Linux 환경의 smoke test 결과

가짜 zone, 균등 배분, 임의 연령 변환은 어댑터에 포함하지 않는다. 전달된 연령 실험 코드와
블록별 CSV도 현재 범위 밖이다. 교산·왕숙·창릉 등은 가능한 서비스 대상의 예시일 뿐이며,
현재 실제 지원 목록이나 신도시별 zone 수가 확정됐다는 의미가 아니다.

AI 팀의 최신 TODO에는 고정 node 문제 해결이 완료된 것으로 표시돼 있다. 다만 현재
어댑터에 연결된 `feature/manage-version`의 `mae:v7.pth`는 실제 검증 결과 1,137개 node에
고정돼 있다. 가변-node 추론을 연결하려면 해당 모델 코드와 호환 체크포인트를 전달받아
별도로 연동 검증해야 한다.

## 운영 주의사항

- 현재 검증 및 권장 device는 CPU다. CUDA/MPS와 GPU 동시 요청은 검증하지 않았다.
- predictor는 각 Gunicorn worker 프로세스 안에서 최초 생성하는 것을 권장한다.
- `preload_app=True` 또는 `--preload` 사용 시 fork 이전에 predictor를 warm-up하지 않는다.
- 체크포인트와 LightGBM은 predictor 생성 시 각각 한 번 로딩되고 이후 요청에서 재사용된다.
- Gunicorn worker가 4개면 모델도 프로세스별로 4번 로딩된다. 실제 서버에서 메모리를
  확인해야 한다.
- Linux는 LightGBM in-process 경로를 사용하지만 이번 macOS 환경에서 검증 완료로
  간주하지 않는다.
- Linux 배포 전 체크포인트·LightGBM 로딩, 합성 forward, 대각 교체, 반복 호출, 프로세스
  메모리를 smoke test한다.
- macOS에서는 PyTorch와 pip LightGBM의 OpenMP 충돌을 피하려고 predictor당 장기
  LightGBM subprocess worker 하나를 재사용한다.
- macOS worker를 fork 이전에 만들면 파이프가 여러 프로세스에 공유될 수 있다.
- macOS 호환 worker의 응답 timeout·자동 재시작은 현재 구현 범위에 포함되지 않는다.
- 안전한 v7 추론을 위해 PyTorch MHA fast path를 프로세스 단위로 비활성화한다. 같은
  프로세스의 다른 MHA 기반 PyTorch 모델에도 영향을 줄 수 있다.

## 기술 참고

### 검증 기준과 환경

- 모델 기준: `feature/manage-version`
- 체크포인트: `best_model/mae:v7.pth`
- strict loading: 74개 state dict key 전부 일치
- 실제 합성 forward: 유한한 `(1, 1137, 1137)` 출력
- 검증 환경: macOS arm64, Python 3.12.12, PyTorch 2.13.0, LightGBM 4.7.0,
  NumPy 2.5.1

검증 버전은 실행 확인 정보이며 requirements에서 동일 버전을 강제로 고정하지 않는다.

### 책임 경계와 모델 교체

`MAEPredictor`는 요청 검증, 외부 전처리기 호출, OD 필터와 JSON 응답을 맡는다. 내부
`_TorchMAERunner`는 체크포인트/LightGBM 1회 로딩, 모델 생성과 tensor forward를 맡는다.

- 모델 구조나 checkpoint 형식 변경: runner 수정
- feature 순서나 정규화 변경: preprocessor 수정
- 출력 의미 변경: postprocessing 수정
- Django `predict()` 계약과 반환 스키마: 가능하면 유지

모든 미래 모델이 자동 호환된다는 의미는 아니다.

### 학습 argument 분류와 근거

| argument | 적용 근거 | 분류 | 추론에서의 의미 |
|---|---|---|---|
| `od_embed_layers=3` | state dict shape | 모델 구조·strict 호환 | `od_embed`의 Linear 3개 구조 |
| `use_self_loop_predictor=True` | state dict key | 모델 구조·strict 호환 | static embedding 기반 대각 MLP 포함 |
| `use_mask_channel=True` | `od_embed` 입력 shape | 모델 구조·전처리 호환 | OD embedding 입력이 `3N` |
| `use_friction=False` | AI 팀 전달 v7 학습 설정 | 모델 forward 동작 | 마지막 `distance_friction` 가산만 끔 |
| `use_lgbm_self_loop=True` | AI 팀 전달 설정·별도 파일 | 추론 후처리 | 최종 대각을 LightGBM으로 교체 |
| `epochs=70` | AI 팀 전달 설정 | 학습 전용 | 추론·생성자에 전달하지 않음 |
| `batch_size=32` | AI 팀 전달 설정 | 학습 전용 | 서비스는 batch 1 실행 |
| `loss_type=weighted_mse` | AI 팀 전달 설정 | 학습 loss | 추론·생성자에 전달하지 않음 |
| `lambda_diag=-1.0` | AI 팀 전달 설정 | 학습 loss | 대각 분리 없이 loss 한 번 계산; 추론 영향 없음 |
| `use_wandb=True` | AI 팀 전달 설정 | 기록 전용 | W&B 기록만 제어 |

`use_friction=False`는 state dict shape로 판별되는 값이 아니다. 생성자는 설정과 관계없이
`distance_friction` parameter를 등록하므로 AI 팀이 전달한 `mae:v7` 학습 설정을 기준으로
적용했다. 이를 `True`로 추정해서 바꾸지 않는다.

### 모델 구조와 tensor

- node 수: 1,137
- static feature 수: 15
- `d_model=128`, attention head 8개, Transformer 4층
- OD embedding: Linear 3개
- `feature_embed.0.weight`: `(128, 15)`
- `od_embed.0.weight`: `(256, 3411)` = `3×1137`
- `decoder.2.weight`: `(2274, 128)` = `2×1137`

입력은 batch 전 `(N,15)`, `(N,N)`, `(N,N)`, `(N,)`이고 runner에서 batch 1로 바꾼다.
출력은 전체 `(1,N,N)`이며 원본 OD target은 `귀가`, `출근`, `등교`, `업무`, `기타`
통행량의 합이다. 학습 log 출력에는 `expm1`을 적용한다.

`od_embed` 입력과 decoder 출력이 N에 직접 고정되므로 1,136개와 1,138개 모델에는 strict
loading이 shape mismatch로 실패한다. 이는 현재 연결된 v7 코드·체크포인트에 대한 결과다.
N이 바뀌는 시나리오에는 AI 팀의 가변-node 지원 모델 코드와 호환 체크포인트를 전달받아
별도로 연동 검증해야 한다.

### 거리, friction과 MHA

`use_friction=False`여도 `x_dist`는 제외되지 않는다. 모델은 `log1p` 거리로 50개 bin을
만들어 learned `distance_bias`를 Transformer attention에 항상 적용한다. 꺼지는 것은 최종
OD 행렬에 더하는 별도 `distance_friction` 항뿐이다.

최신 PyTorch native MHA fast path는 이 모델의 head별 3차원 learned attention mask와
eval/inference mode에서 NaN을 만들었다. runner는 model을 `eval()`로 유지하고 fast path를
프로세스 단위로 비활성화한다. 안전 경로에서 실제 v7 합성 forward가 성공했다.

### mask channel

`use_mask_channel=True`이면 `(B,N)` bool mask를 float로 바꾸고 `unsqueeze(1)` 후
`(B,N,N)`로 확장해 OD row/column 뒤에 붙인다. 마스킹 zone의 OD row와 column은 외부
전처리에서 0이어야 하며, 해당 node의 OD embedding은 모델 안에서 mask token으로 바뀐다.

### static feature와 StandardScaler

feature 순서는 다음과 같다.

```text
station_count_준고속철도
행정동전체면적_m2
상업업무지역비율_pct
공공시설지역비율_pct
주거지역비율_pct
station_count_고속철도
pop_0_19
pop_20_59
pop_60_plus
worker_count
business_count
아파트비율_퍼센트
station_count_지하철
station_count_일반철도
is_masked
```

원본 14개 feature는 test node를 제외한 train node에 `StandardScaler`를 fit해 변환한 뒤
`is_masked`를 붙였다. 마스킹 node는 변환된 `worker_count`, `business_count`를 0,
`is_masked`를 1로 만든다. scaler 평균·표준편차는 체크포인트나 별도 파일에 저장돼 있지
않다.

API의 `0–19 / 20–39 / 40–64 / 65+`를 모델의
`pop_0_19 / pop_20_59 / pop_60_plus`로 바꾸는 공식 규칙도 아직 없다. 전처리기는 입력
비율을 임의 변환하거나 다른 연령 예측으로 조용히 덮어쓰면 안 된다.

### node index

기존 행정동 matrix index는 `dataset/raw/OD_dong_list.xlsx`의 `dong_code` 행 순서를 0부터
센 값이다. 실제 파일에는 중복 없는 1,137개가 있고 `11010530→0`, `11010540→1`,
`31380410→1136`이다. 새 전체 순서는 origin/destination code와 모든 행렬 축에 동일하게
적용해야 한다.

### LightGBM 대각 교체

GNN의 `use_self_loop_predictor=True` MLP가 먼저 대각 보정값을 더한다. 이어서 v7 정책은
실수 단위 행렬의 전체 대각을 `best_lgbm_self_loop.txt` 예측으로 덮는다. 따라서 GNN
체크포인트만의 forward와 완전한 v7 추론 정책은 구분해야 한다.

macOS worker와 Linux in-process 경로 모두 LightGBM 호출 직전에 입력을 CPU
`float32`로 통일한다. LightGBM 모델은 predictor 생성 시 한 번만 로딩된다.

### 폴더만 별도 전달할 때

`mae_backend_adapter/`만으로는 기본 runner를 실행할 수 없다. 다음 구조를 함께 전달하고
`handoff-root`를 Python import path에 둔다. 작업 디렉터리가 다르면 `model_path`에
절대경로를 전달한다.

```text
handoff-root/
├── mae_backend_adapter/
├── best_model/
│   ├── mae:v7.pth
│   └── best_lgbm_self_loop.txt
└── src/
    └── mae/
        └── models.py
```

기본 runner가 import하는 다른 `src/mae` 파일은 없다. LightGBM 기본 경로는
`model_path.with_name("best_lgbm_self_loop.txt")`이므로 두 모델 파일을 다른 디렉터리에
둘 때는 `lgbm_model_path`를 명시한다.
