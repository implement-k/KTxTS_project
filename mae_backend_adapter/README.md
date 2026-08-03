# mae-year 백엔드 Adapter

## 1. 패키지가 하는 일

`mae_backend_adapter`는 Django 백엔드의 Mock Provider 자리에 최종 MAE 모델을 연결한다.
모델은 process마다 한 번 로딩하고 요청마다 재사용한다.

호출 경계는 다음과 같다.

| 경계 | 입력 |
|---|---|
| 공개 Django API | `newtown_code`, `total_population`, `age_ratios` |
| 백엔드 `PredictionRequest` | `newtown_code`, `newtown_name`, `total_population`, `age_ratios` |
| 내부 MAE Predictor | `newtown=<DB의 정식 이름>`, `total_population`, `age_ratios` |

Adapter raw 결과는 신도시 관련 OD 목록이다. 전체 OD 행렬은 API 응답에 포함하지
않는다. 시간대별 예측은 지원하지 않으며 현재 결과는 **일평균 이동량**이다.

### Node 수 `N`

현재 API는 도시 한 곳을 요청한다. 서버가 `newtown_code`에 맞게 선택한
도시별 모델 입력 데이터 묶음(artifact)의
`ModelInputs.x_static.shape[0]`이 `N`을 결정한다. `N`은 API 요청값이 아니며 특정 node
수를 가정하지 않는다.

가상 node 수가 달라져도 Adapter 수정은 필요 없다. 전체 도시 node의 합집합 데이터도
구조상 가능하지만, 현재 연동에서는 도시별 데이터 묶음 방식을 권장한다.

## 2. 백엔드가 준비할 데이터

역할을 다음과 같이 나눈다.

| 담당 | 제공·구현 항목 |
|---|---|
| AI 팀 | 모델 코드·checkpoint·LightGBM, feature schema, canonical node code·순서, OD·거리·인접 데이터 또는 생성 규칙, mask 규칙, scaler 재현 정보, model/input-data version·checksum |
| 백엔드 | `newtown_code`에 맞는 데이터 묶음 선택, Django settings, concrete `PopulationPreprocessor`, `PredictionRequest` wrapper, DB `ModelVersion`/`ModelNode`, cache·실행 기록·Matrix·지도·인사이트 |

교산·창릉·왕숙 경계·가상 node 구성과 인구·static feature 배분은 경훈님이
작업 중이다. 도시별 기본 연령 비율도 경훈님의 예측 모델 결과로 추후 제공할
예정이다. 초기·중기·완료 인구는 AI 팀이 앞으로 정해야 하며, static feature
기본값과 가상 node별 배분값도 미정이다. 프론트의 더미 값은 모델 운영 기본값이 아니며,
Adapter와 운영 전처리기는 더미 값으로 자동 추론하지 않는다.

필수 데이터나 설정이 없으면 합성값을 만들지 않고
`PreprocessingConfigurationError`를 발생시킨다.

## 3. 빠른 연결 방법

의존성을 설치한다.

```bash
python -m pip install -r mae_backend_adapter/requirements.txt
```

각 Django worker에서 Predictor를 한 번 만들고, 백엔드 `PredictionRequest`를 명시적으로
변환한다.

```python
from functools import lru_cache

from django.conf import settings
from mae_backend_adapter import MAEProvider
from project.inference.preprocessing import project_preprocessor


@lru_cache(maxsize=1)
def get_od_provider():
    return MAEProvider(
        model_path=settings.MAE_MODEL_PATH,
        lgbm_model_path=settings.MAE_LGBM_MODEL_PATH,
        model_module_path=settings.MAE_MODEL_MODULE_PATH,
        device=settings.MAE_DEVICE,
        preprocessor=project_preprocessor,
    )


def predict_od(request):
    predictor = get_od_provider()
    result = predictor.predict(
        newtown=request.newtown_name,
        total_population=request.total_population,
        age_ratios={
            key: float(value)
            for key, value in request.age_ratios.items()
        },
    )
    return result
```

실제 Django Provider 위치, settings와 버전 매핑은
[`docs/backend_integration.md`](docs/backend_integration.md)를 참고한다.

## 4. 입력 DTO

공개 Django API 요청은 도시 이름이 아닌 `newtown_code`를 받는다.

```json
{
  "newtown_code": "gyosan",
  "total_population": 100000,
  "age_ratios": {
    "0_19": 0.2,
    "20_59": 0.6,
    "60_plus": 0.2
  }
}
```

백엔드가 code를 DB의 정식 이름으로 조회한 뒤 내부 Predictor를 호출한다.
`PopulationPreprocessor`는 이 요청과 선택된 도시별 데이터 묶음을 전처리 완료
`ModelInputs`로
변환한다.

| 필드 | Shape / dtype | 전달할 값 |
|---|---|---|
| `x_static` | `(N, F)` floating | 학습 기준 scaler와 feature 정렬이 적용된 값. 현재 checkpoint의 `F=20` |
| `x_od_masked` | `(N, N)` floating | `log1p`와 mask 처리가 끝난 OD |
| `x_dist` | `(N, N)` floating | `log1p`가 적용된 거리 |
| `a_spatial` | `(N, N)` floating | 공간 인접 0/1 행렬 |
| `mask` | `(N,)` bool | 현재 운영은 선택 신도시 node만 `True` |
| `active_node_mask` | `(N,)` bool | 현재 운영은 모두 `True`; 병합·삭제 node는 `False` 지원 |
| `origin_codes` | 길이 `N` | 행 순서의 node code |
| `destination_codes` | 길이 `N` | 열 순서의 node code. origin과 같은 순서 |
| `newtown_zone_codes` | code 목록 | 전체 node code에 포함된 신도시 node |

원본 OD·거리 파일은 code로 재정렬할 수 있어 파일 행 순서를 고정할 필요가 없다.
하지만 `ModelInputs`에서는 한 canonical node code 목록을 기준으로 static feature,
OD·거리·인접 행렬의 두 축, 두 mask와 origin/destination code 순서가 모두 같아야
한다. code 기반 reindex는 concrete `PopulationPreprocessor`의 책임이다.

Adapter는 `x_static.shape[0]`에서 `N`을 결정하고 shape와 code 길이를 검증하지만,
tensor의 의미적 순서가 맞는지는 알 수 없다. 원본 static feature column 순서는 달라도
되지만, 18개 feature 이름은 정확히 모두 제공해야 한다. helper가 이름을 기준으로
artifact의 canonical feature 순서로 재정렬하며, 최종 `ModelInputs.x_static`은 이
canonical 순서의 18개 feature 뒤에 indicator 2개가 붙은 순서여야 한다.

### 2023 static scaler

`load_static_scaler()`는 NPZ·JSON을 process 당 한 번 로드하고 같은 객체를 재사용한다.
요청마다 scaler를 fit하지 말고, 기존 node와 신도시 가상 node의 raw static feature에
모두 같은 2023 scaler를 적용한다. `is_masked`, `is_merged`는 scaling에서 제외하고
18개 feature를 변환한 뒤 `is_masked == 1`인 행의 masking feature 4개를 scaling 공간의
0으로 설정하고 두 indicator를 이 순서로 붙인다. `is_merged`만 1인 행은 mask하지 않는다.
scaler 해시, feature 누락·추가·중복, metadata의 masking 이름·index 또는 입력 차원이
맞지 않으면 임의 값으로 대체하지 않고 예외로 실패한다.

> 주의: 이 scaler는 현재 2023 Dataset과 train/val/test split으로 복원한 값이며 현재
> `ODDataset.scaler` 및 `X_static`과 수치적으로 일치한다. checkpoint 내부에는 scaler가
> 없어 학습 당시 값과 역사적으로 동일하다는 직접 기록은 없다. 저장소 이력과 checkpoint
> provenance를 근거로 현재 2023 scaler를 호환 scaler로 채택했다.

```python
from mae_backend_adapter import load_static_scaler

static_scaler = load_static_scaler()  # server/worker 시작 시 한 번
x_static = static_scaler.transform_with_indicators(
    raw_values,
    feature_names=raw_feature_names,
    is_masked=is_masked,
    is_merged=is_merged,
)
```

현재 운영 규칙에서 mask된 신도시 node의 OD 행·열은 0이고 해당 node는
active 상태여야 한다. 전처리·feature 상세는
[`docs/model_contract.md`](docs/model_contract.md)를 참고한다.

공개 `predict()`는 항상 명시적인 `total_population`과 `age_ratios`를 요구한다.

Ablation 설정은 학습·평가 실험 설정이며 백엔드 런타임 요청이 아니다.

## 5. 응답 예시

다음은 **Adapter raw 응답**이다. Django `/predict/`와 `/analyze/` 응답은 백엔드
DTO와 service 계층이 별도로 만든다.

```json
{
  "newtown": "교산",
  "newtown_zone_codes": ["VIRTUAL_GYOSAN"],
  "od": [
    {
      "origin_code": "VIRTUAL_GYOSAN",
      "destination_code": "11010530",
      "predicted_trips": 123.5,
      "movement_type": "outflow"
    },
    {
      "origin_code": "11010530",
      "destination_code": "VIRTUAL_GYOSAN",
      "predicted_trips": 98.25,
      "movement_type": "inflow"
    }
  ],
  "metadata": {
    "model_family": "mae-year/ODMAE",
    "model_version": "mae:hybrid-86epoch",
    "checkpoint": "mae:hybrid-86epoch.pth",
    "output_transform": "log1p",
    "self_loop_policy": "lightgbm_override"
  }
}
```

`movement_type`은 `internal`, `outflow`, `inflow` 중 하나다. `predicted_trips`는 반올림하지
않는 실수다. `metadata.model_version`은 checkpoint 파일명에서 나온 값으로 Django
`ModelVersion.code`와 다를 수 있다. 백엔드는 별도의 안정적인 deployment version을
설정하고 응답 metadata를 정규화해야 한다.

## 6. 후처리 정책

Provider가 보장하는 정책은 다음과 같다.

- 음수 이동량은 0으로 보정
- 동일 origin/destination code는 합산
- 합산 결과가 0인 OD는 제외
- 외부→외부 OD는 제외
- 누락 node code는 `UNMAPPED`
- inactive node는 결과에서 제외

Matrix, 지도 node·이동선과 교통 인사이트는 Django service 계층이 계산한다.

- 유입 우세: 유입/유출 ≥ 1.2
- 유출 우세: 유입/유출 ≤ 0.8
- 균형: 0.8 < 유입/유출 < 1.2
- 권역 집중: Top 1 권역 비중 ≥ 40%
- 내부 이동 중심: 내부 이동 비중 ≥ 50%

## 7. 테스트와 현재 제한사항

저장소 root에서 실행한다.

```bash
python -m unittest discover -s mae_backend_adapter/tests -v
```

테스트는 다음을 확인한다.

- 같은 Adapter가 서로 다른 작은 `N`을 연속 처리
- 모든 `(N, N)` tensor, `(N,)` mask와 code 길이 검증
- 요청 검증, checkpoint strict load와 후처리 계약
- 저장소의 2023 `ODDataset` sample과 실제 checkpoint·LightGBM을 사용한 CPU 구조 smoke

현재 smoke는 운영 교산·창릉·왕숙 추론 검증이 아니다. 운영용 도시별 데이터
묶음과 concrete `PopulationPreprocessor`를 연결하기 전에는 고수준 `predict()`를
실행할 수 없다.

추가 제한사항은 다음과 같다.

- CUDA/MPS는 배포 장비에서 추가 검증이 필요하다.
- 추론 memory는 `N²`에 비례하므로 worker 수와 동시 요청 수를 부하 시험해야 한다.
- 추적된 일부 fixed-eval 데이터는 최신 전처리 schema와 달라 재생성이 필요하다.

모델 상세는 [`docs/model_contract.md`](docs/model_contract.md), Django 연결은
[`docs/backend_integration.md`](docs/backend_integration.md)를 참고한다.
