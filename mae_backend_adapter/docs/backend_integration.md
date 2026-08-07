# 백엔드 연결 안내

## 전달 범위

백엔드는 `mae_backend_adapter/` 폴더 전체를 같은 구조로 배포해야 한다. 다음 항목이
모두 패키지 내부 상대 경로로 연결된다.

- `model/mae.py`, `model/mae.pth`
- `artifacts/mae_year_2023_static_scaler.json`, `.npz`
- `newtown/` 아래 도시별 입력 데이터

외부 MAE 모델 경로, LightGBM 경로와 active mask 설정은 필요하지 않다.

## 권장 loader

Django worker 등 서버 process마다 predictor를 한 번만 만든다.

```python
from functools import lru_cache

from mae_backend_adapter import NewtownPredictor


@lru_cache(maxsize=1)
def get_mae_predictor() -> NewtownPredictor:
    return NewtownPredictor(device="cpu")
```

요청 처리부에서는 백엔드의 신도시 코드를 그대로 전달한다.

```python
result = get_mae_predictor().predict(
    request.newtown_code,       # all, changneung, gyosan, wangsuk
    period=request.period,      # initial, middle, final
)
```

`period`를 전달하지 않으면 `final`이 기본값이다. 백엔드의 한글 표시 이름을 모델에
넘기지 않는다.

## 오류 처리

- 지원하지 않는 도시·시기: `InputValidationError`
- 번들 데이터 또는 scaler 문제: `PreprocessingConfigurationError` 또는
  `StaticScalerArtifactError`
- checkpoint 또는 모델 구조 불일치: `CheckpointLoadError` 또는
  `CheckpointCompatibilityError`
- tensor shape·dtype 문제: `TensorShapeError`

필수 파일이 없거나 형식이 맞지 않으면 합성값으로 대체하지 않고 명시적으로 실패한다.

## 응답 연결

Adapter의 `od`는 신도시 관련 OD만 포함한다.

```json
{
  "newtown": "gyosan",
  "newtown_zone_codes": ["50000001"],
  "od": [
    {
      "origin_code": "50000001",
      "destination_code": "31180510",
      "predicted_trips": 123.4,
      "movement_type": "outflow"
    }
  ],
  "metadata": {
    "period": "final",
    "checkpoint": "mae.pth"
  }
}
```

백엔드는 이 결과를 서비스 응답 DTO에 맞게 변환하면 된다. 모델 출력의 전체 OD 행렬은
공개 결과에 포함하지 않는다.

## 배포 전 확인

```bash
python -m unittest discover -s mae_backend_adapter/tests -v

MAE_RUN_REAL_NEWTOWN_SMOKE=1 \
python -m unittest mae_backend_adapter.tests.test_real_newtown_smoke -v
```

실제 smoke는 `all`, `changneung`, `gyosan`, `wangsuk`과 세 period를 번들
checkpoint로 실행한다. 백엔드 저장소에서는 같은 테스트를 handoff 경로에 맞춰 한 번
실행한 뒤 서비스 계층 테스트를 추가하면 된다.
