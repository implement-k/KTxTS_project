# mae-year 모델 상세 계약

이 문서는 Adapter 유지보수에 필요한 모델 내부 계약을 설명한다. 일반적인 백엔드 연결은
상위 [`README.md`](../README.md)만 읽으면 된다.

## 1. 동적 node 수

Adapter는 `ModelInputs.x_static.shape`을 읽어 다음과 같이 `N`과 feature 수를 정한다.

```python
node_count, feature_count = inputs.x_static.shape
```

이 값으로 다음 조건을 검증한다.

- `x_od_masked`, `x_dist`, `a_spatial`: `(N, N)`
- `mask`, `active_node_mask`: `(N,)`
- `origin_codes`, `destination_codes`: 길이 `N`
- 모델 출력: `(1, N, N)` 또는 `(N, N)`

Checkpoint, 환경 변수와 공개 요청에서 `N`을 읽지 않는다. Decoder도 node 수에 독립적이다.
도시별 모델 입력 데이터 묶음의 node 구성이 변해도 Adapter 수정은 필요 없다.

원본 OD·거리 파일은 행·열 code로 reindex할 수 있으므로 파일 순서가 고정일
필요는 없다. 다만 tensor에는 code 라벨이 없다. concrete `PopulationPreprocessor`는
하나의 canonical node code 목록을 기준으로 다음을 모두 재정렬해야 한다.

- static feature 행
- OD·거리·공간 인접 행렬의 행과 열
- `mask`, `active_node_mask`
- `origin_codes`, `destination_codes`

Adapter는 shape, dtype과 code 길이를 검증하지만 의미적 node 순서는 판별할 수 없다.
code 기반 reindex와 순서 보장은 concrete `PopulationPreprocessor`의 책임이다.

`MAEPredictor.num_nodes`는 이전 호출부 호환을 위해 남긴 속성이며 값은 `None`이다. 실제
요청의 node 수는 응답 `metadata.node_count`에 기록한다.

## 2. 모델 입력과 전처리

`ODMAE.forward()` 인자 순서는 다음과 같다.

```python
pred_log = model(
    x_static,
    x_od_masked,
    x_dist,
    a_spatial,
    mask,
    active_node_mask,
)
```

Runner가 batch 차원을 추가한 뒤의 계약이다.

| 필드 | Shape | dtype | 의미 |
|---|---:|---|---|
| `x_static` | `(B, N, F)` | `float32` | 표준화된 static feature와 indicator |
| `x_od_masked` | `(B, N, N)` | `float32` | `log1p` OD, mask·inactive 행/열은 0 |
| `x_dist` | `(B, N, N)` | `float32` | `log1p` 거리 |
| `a_spatial` | `(B, N, N)` | `float32` | 공간 인접 0/1 행렬 |
| `mask` | `(B, N)` | `bool` | `True`가 예측 대상 |
| `active_node_mask` | `(B, N)` | `bool` | 병합·삭제 node는 `False` |

현재 checkpoint는 18개 static feature에 `is_masked`, `is_merged`를 붙인 `F=20`을 사용한다.
Dataset은 `dong_code`, `dong_name`을 제외한 column 이름을 정렬한 뒤 tensor로
변환한다. 원본 CSV column 순서는 달라도 되지만 운영 전처리기는 다음과 같은
정렬된 feature 이름 목록을 사용해야 한다. 모델 자체는 feature 이름을 받거나
정렬하지 않고 이미 정렬된 `(N, F)` tensor를 받는다.

```text
business_count
business_density
pop_0_19
pop_20_59
pop_60_plus
station_count_고속철도
station_count_일반철도
station_count_준고속철도
station_count_지하철
station_density_지하철
worker_count
worker_density
공공시설지역비율_pct
기타지역비율_pct
상업업무지역비율_pct
아파트비율_퍼센트
주거지역비율_pct
행정동전체면적_m2
```

Dataset의 scaler 처리는 다음과 같다.

```python
scaler.fit(raw_static[train_indices])
scaler.transform(raw_static)
```

운영 요청이나 가상 신도시 입력으로 scaler를 새로 fit하면 안 된다.
`mae_backend_adapter/artifacts/mae_year_2023_static_scaler.{npz,json}`을 process 당
한 번 로드해 기존 node와 가상 node에 같이 사용한다.

> 주의: 이 scaler는 현재 2023 Dataset과 split으로 복원한 값이며,
> 현재 `ODDataset.scaler` 및 `X_static`과 수치적으로 일치한다. checkpoint에는 scaler가
> 들어 있지 않아 학습 당시 값과 동일하다는 직접 기록은 없다. 저장소 이력과 checkpoint
> 정보를 근거로 호환 scaler로 채택했다.

`StaticFeatureScaler.transform_with_indicators()`의 처리 순서는 다음과 같다.

1. feature 이름을 기준으로 artifact의 canonical 순서로 재정렬
2. 저장된 2023 scaler 적용
3. `is_masked == 1`인 행의 `worker_count`, `business_count`, `worker_density`,
   `business_density`를 scaling 공간에서 0 처리
4. `is_masked`, `is_merged` indicator를 이 순서로 추가

`is_merged`만 1인 행은 masking feature를 0으로 만들지 않는다. Artifact 해시나 feature
schema가 맞지 않으면 대체값 없이 실패한다.

현재 운영 추론의 mask 규칙은 다음과 같다.

- `active_node_mask`: 모든 node가 `True`
- `mask`: 선택한 신도시 node만 `True`
- mask된 node의 OD 행과 열은 0
- 신도시 node는 active 상태

Adapter의 일반 계약은 병합·삭제 node를 위한 `active_node_mask=False`도 계속 지원한다.
Mask 대상 node의 `worker_count`, `business_count`, `worker_density`,
`business_density`를 0으로 만들고 `is_masked=1`로 설정하는 학습 전처리도 유지한다.

다음 운영 기본값은 아직 미정이며 Adapter에 하드코딩하지 않는다.

- 초기·중기·완료 인구
- 도시별 기본 연령 비율
- static feature 기본값
- 가상 node별 인구·연령 배분값

경계·가상 node 구성과 인구·static feature 배분은 경훈님이 작업 중이다. 도시별
기본 연령 비율도 경훈님의 예측 모델 결과로 추후 제공할 예정이다. 단계별 인구는
AI 팀이 앞으로 정해야 한다. 프론트의 더미 값은 운영 기본값이 아니며, Adapter와 전처리기는
더미 값으로 자동 추론하지 않는다. `predict()`는 명시적인 `total_population`과
`age_ratios`를 계속 요구한다.

## 3. 출력

모델 출력은 `float32 (B, N, N)`의 `log1p` 공간 tensor다. Provider는 다음 순서로
서비스 OD를 만든다.

1. `torch.expm1()` 적용
2. NaN·무한대 거부
3. 음수 0 보정
4. 설정된 경우 LightGBM self-loop로 대각 교체
5. inactive node 제외
6. 신도시 관련 OD 선택
7. 중복 code 합산과 0 OD 제외

모델은 시간대 feature를 입력받지 않는다. 출력 단위는 일평균 이동량이다.

## 4. Checkpoint 구조 판별

Loader는 raw `state_dict`와 `state_dict` 또는 `model_state_dict` wrapper를 지원한다.
`weights_only=True`, CPU map location으로 읽은 뒤 다음 state shape를 사용한다.

- `feature_embed.0.weight`: `d_model`, `num_features`
- `distance_bias.weight`: attention head 수
- `transformer.layers.*`: transformer layer 수
- `self_loop_predictor.*`: MAE self-loop predictor 포함 여부

`src/mae-year/models.py::ODMAE`를 생성한 후 `strict=True`로 load한다. 현재 배포 설정은
distance friction을 사용하지 않는다. Ablation option은 백엔드 요청이 아니라 학습·평가
실험 설정이다.

## 5. LightGBM 내부 실행

최종 평가 설정은 `best_lgbm_self_loop.txt`의 log 예측으로 MAE 출력 대각을 교체한다.
LightGBM 입력은 같은 node 순서의 전처리 완료 `x_static`이며 `float32` CPU tensor다.

- macOS: PyTorch와 OpenMP 충돌을 피하기 위해 별도 worker process에서 실행
- 그 외 플랫폼: process 내부 `lightgbm.Booster` 사용
- 테스트 주입: `self_loop_predictor` callable로 교체 가능

MAE 대각을 그대로 사용할 배포는 `use_lgbm_self_loop=False`를 명시한다.

## 6. 과거 v7 Adapter와의 차이

| 구분 | 과거 v7 | 현재 mae-year |
|---|---|---|
| 모델 class | `SpatialODMAE` | `ODMAE` |
| Forward | 4개 tensor | 6개 tensor |
| Static feature | 15 | 20 |
| Node 수 | decoder에서 고정값 추론 | `x_static.shape[0]`에서 요청마다 결정 |
| 추가 입력 | 없음 | `a_spatial`, `active_node_mask` |
| 중복 code | 거부 | OD 합산 |
| inactive node | 입력 없음 | 입력 검증 및 결과 제외 |

공개 `predict(newtown, total_population, age_ratios)`와 응답 top-level 구조는 유지한다.

## 7. 알려진 입력 데이터 제한

추적된 일부 `dataset/fixed_eval/base_data_*.pt`는 최신 전처리 코드가 요구하는
`X_dist`, `A_spatial` key가 없다. 실제 smoke는 현재 `ODDataset`이 직접 만든 입력으로
수행한다. fixed-eval 데이터를 백엔드에서 재사용하려면 최신 schema로 다시 생성해야 한다.
