# MAE 모델·전처리 계약

## 배포 모델

배포 기준은 다음 번들 파일 한 쌍이다.

- 구조: `mae_backend_adapter/model/mae.py`
- checkpoint: `mae_backend_adapter/model/mae.pth`

Runner는 checkpoint의 feature 차원, attention head와 transformer layer 수를 읽어
`ODMAE`를 만들고 `strict=True`로 로드한다. 외부 모델 source와 checkpoint를
주입하지 않는다.

## 입력 tensor

`N`은 `x_static.shape[0]`에서 요청마다 결정되며 고정 설정값이 아니다.

| 필드 | Adapter 입력 | 모델 입력 | 의미 |
|---|---:|---:|---|
| `x_static` | `(N, 20)` | `(1, N, 20)` float32 | 18개 scaled feature + indicator 2개 |
| `x_od_masked` | `(N, N)` | `(1, N, N)` float32 | `log1p` OD, mask 행·열은 0 |
| `x_dist` | `(N, N)` | `(1, N, N)` float32 | `log1p` 거리 |
| `a_spatial` | `(N, N)` | `(1, N, N)` float32 | 0/1 공간 인접 행렬 |
| `mask` | `(N,)` | `(1, N)` bool | 선택 신도시 node가 `True` |

`city_codes`는 위 tensor와 동일한 canonical node 순서의 길이 `N` 목록이다.
`newtown_zone_codes`는 `city_codes` 안에 존재해야 하며 모두 `mask=True`여야 한다.

모든 node를 사용하므로 `active_node_mask`는 없다. 모델은 대각을 포함한 전체 OD를
직접 예측하며 별도 self-loop 모델이나 LightGBM 보정을 사용하지 않는다.

## 신도시 데이터 전처리

`NewtownPreprocessor`는 다음 순서로 입력을 만든다.

1. `OD_dong_list_2023.xlsx`의 `dong_code`를 canonical 순서로 사용
2. long-format `dong_distance.csv`를 같은 순서의 정방 행렬로 변환하고 `log1p`
3. `dong_adjacency.pkl`을 같은 순서의 0/1 정방 행렬로 변환
4. 선택 period의 `static_features_{period}.csv`를 canonical 순서로 재정렬
5. `mask_code.json`의 신도시 node를 mask
6. `newtown/od_data_2023.csv`를 canonical OD 행렬로 집계하고 `log1p`
7. mask node가 출발지 또는 도착지인 OD를 0으로 설정
8. 번들 scaler로 static feature를 변환하고 indicator를 추가

지원 도시는 `all`, `changneung`, `gyosan`, `wangsuk`, 지원 period는
`initial`, `middle`, `final`이다.

## Static scaler

`dong_code`, `dong_name`을 제외한 18개 feature 이름을 정렬해 artifact schema와
일치시킨다. `StandardScaler`는 학습 train node로 fit된 저장 parameter를 사용하며
요청 데이터로 다시 fit하지 않는다.

Mask node는 scaled 결과의 다음 네 feature를 0으로 만든다.

- `worker_count` (`10`)
- `business_count` (`0`)
- `worker_density` (`11`)
- `business_density` (`1`)

그 뒤 `is_masked`, `is_merged`를 이 순서로 붙인다. 현재 신도시 입력에서
`is_merged`는 모두 0이다.

Scaler NPZ는 현재 `src/mae-year/dataset.py`의 2023 `ODDataset` parameter 및
`X_static`과 수치 비교한다. JSON의 target checkpoint는 실제 배포 파일인
`mae_backend_adapter/model/mae.pth`다.

## 출력 처리

모델 출력은 log 공간의 `(N, N)` 행렬이다.

1. `torch.expm1()` 적용
2. NaN·무한대 거부
3. 음수는 0으로 보정
4. 신도시 내부·유입·유출 OD만 선택
5. 중복 code는 합산하고 0 OD는 제외

반환 결과에는 전체 행렬 대신 OD 목록과 실행 metadata가 포함된다.
