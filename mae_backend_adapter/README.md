# MAE 신도시 백엔드 Adapter

`mae_backend_adapter`는 신도시 코드만 받아 번들 데이터를 전처리하고 MAE OD 예측
결과를 반환하는 패키지다. 모델 구조, checkpoint, scaler와 신도시 데이터가 패키지
내부에 포함되어 있어 백엔드가 별도 모델 경로를 주입하지 않는다.

## 빠른 사용법

서버 process마다 `NewtownPredictor`를 한 번 만들고 재사용한다.

```python
from mae_backend_adapter import NewtownPredictor

predictor = NewtownPredictor(device="cpu")

# period를 생략하면 final을 사용한다.
result = predictor.predict("gyosan")
result = predictor.predict("wangsuk", period="initial")
```

지원 도시 코드는 `all`, `changneung`, `gyosan`, `wangsuk`이고 지원 시기는
`initial`, `middle`, `final`이다.

단발성 실행은 다음과 같이 할 수 있다.

```python
from mae_backend_adapter import predict_newtown

result = predict_newtown("gyosan")
```

`predict_newtown()`은 호출할 때마다 모델을 새로 읽으므로 서버에서는
`NewtownPredictor` 재사용을 권장한다.

## 패키지 구성

| 경로 | 역할 |
|---|---|
| `newtown_predictor.py` | 도시·시기 선택, 번들 데이터 전처리, 간단한 예측 API |
| `predictor.py` | 5개 tensor 검증, 모델 실행, OD JSON 변환 |
| `model/mae.py` | 배포용 `ODMAE` 구조 |
| `model/mae.pth` | 배포용 checkpoint |
| `artifacts/` | 2023년 static feature scaler |
| `newtown/` | 도시별 OD 목록·거리·인접·static feature·mask 데이터 |
| `docs/backend_integration.md` | 백엔드 연결 안내 |
| `docs/model_contract.md` | tensor와 전처리 계약 |

모델은 5개 tensor(`x_static`, `x_od_masked`, `x_dist`, `a_spatial`, `mask`)를
입력받는다. 모든 node를 사용하므로 별도 active mask가 없고, 대각을 포함한 전체 OD를
모델이 직접 예측하므로 LightGBM 보정도 없다.

## 반환값

반환 dict의 주요 필드는 다음과 같다.

- `newtown`: 요청 도시 코드
- `newtown_zone_codes`: 선택한 신도시 가상 node 코드
- `od`: 신도시 내부·유입·유출 OD 목록
- `metadata`: period, checkpoint, node 수, feature 수 등 실행 정보

각 `od` 항목에는 `origin_code`, `destination_code`, `predicted_trips`,
`movement_type`이 있다. 외부 지역끼리의 OD와 값이 0인 OD는 제외된다.

## 설치와 테스트

```bash
python -m pip install -r mae_backend_adapter/requirements.txt
python -m unittest discover -s mae_backend_adapter/tests -v
```

실제 번들 데이터와 checkpoint로 모든 도시·시기를 실행하려면 다음 명령을 사용한다.

```bash
MAE_RUN_REAL_NEWTOWN_SMOKE=1 \
python -m unittest mae_backend_adapter.tests.test_real_newtown_smoke -v
```

상세 백엔드 연결 방법은 `docs/backend_integration.md`를 따른다.
