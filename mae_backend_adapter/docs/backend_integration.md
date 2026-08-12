# Django 백엔드 연동

이 문서는 `mae_backend_adapter`를 현재 Django REST Framework 백엔드에 연결할 때
필요한 경계만 정리한다.

## 백엔드 요청 변환

공개 API는 `newtown_code`, `total_population`, `age_ratios`를 받는다. Serializer가
`newtown_code`로 활성 `NewTown`을 찾은 뒤 다음 `PredictionRequest`를 만든다.

```text
newtown_code
newtown_name
total_population
age_ratios
```

Provider wrapper는 이 DTO를 내부 Predictor 인자로 명시적으로 변환한다.

```python
result = predictor.predict(
    newtown=request.newtown_code,
    total_population=request.total_population,
    age_ratios={
        key: float(value)
        for key, value in request.age_ratios.items()
    },
)
```

Serializer의 `validated_data`는 `newtown_code`와 내부용 `new_town` 객체를 포함할 수 있으므로
`predict(**validated_data)`를 사용하지 않는다.

## Provider wrapper 위치

현재 백엔드 wrapper는 `apps/predictions/adapters/mae_v7.py`에 있다. mae-year로
교체할 때의 권장 위치는 다음과 같다.

```text
apps/predictions/adapters/mae_year.py
apps/predictions/integrations/mae_loader.py
apps/predictions/integrations/mae_preprocessor.py
```

`mae_year.py`는 백엔드 `PredictionProvider` 계약과 예외 변환을 담당한다.
`mae_loader.py`는 adapter에 포함된 model·preprocessor를 Django worker process당 한 번 로딩한다.
`mae_preprocessor.py`는 선택된 도시별 모델 입력 데이터 묶음을 `ModelInputs`로 바꾼다.

## Django settings

### 현재 백엔드에 이미 존재하는 설정

| 설정 | 현재 역할 |
|---|---|
| `MAE_HANDOFF_ROOT` | Adapter 배포 root |
| `MAE_DEVICE` | `cpu`, `cuda` 등 PyTorch device |
| `PREDICTION_BACKEND` | Provider 선택값. 현재 factory는 `mock`, `mae_v7`만 지원 |
| `PREDICTION_CACHE_VERSION` | `/analyze/` 결과 cache namespace version |

### mae-year 연결 시 추가하거나 변경할 설정

| 설정 | 필요한 결정·변경 |
|---|---|
| `MAE_MODEL_PATH`, `MAE_LGBM_MODEL_PATH` | 모델이 adapter 내부에 포함되므로 제거 |
| `MAE_SUPPORTED_NEWTOWNS` | adapter의 code 목록을 사용하므로 제거 |
| `PREDICTION_BACKEND` | factory에 `mae_year` 선택값을 구현한 뒤 해당 값으로 변경 |
| `PREDICTION_CACHE_VERSION` | model·input-data 또는 결과 규칙 변경을 반영해 갱신 |
| `MAE_MODEL_VERSION_CODE` | 현재는 없음. 명시적 버전 매핑에 설정을 쓸지 활성 DB version을 조회할지 결정 |

### 도시별 운영 데이터 확정 후 추가할 설정

| 설정 후보 | 결정할 내용 |
|---|---|
| `MAE_PREPROCESSING_ARTIFACT_ROOT` | 현재는 없음. 도시별 입력 데이터 저장 위치와 선택 규칙 |
| `MAE_ARTIFACT_MANIFEST_PATH` | 현재는 없음. 운영 데이터 형식 확정 후 manifest 이름·구조·필요 여부 |

도시별 전체 manifest 파일은 아직 만들어지지 않았으며 현재 필수 전달 파일이 아니다.
운영 데이터 형식이 확정된 뒤 AI·백엔드 팀이 이름과 구조를 함께 결정한다.

## 백엔드가 수정할 항목

- loader에서 외부 model source·checkpoint·LightGBM 경로 검증 제거
- `MAEPredictor(device=..., preprocessor=...)`로 process당 한 번 생성
- `mae_year` Provider 선택값을 현재 `mock`, `mae_v7`만 지원하는 factory에 추가
- `TensorShapeError`를 `InputValidationError`보다 먼저 예외 변환
- concrete `PopulationPreprocessor`를 loader에 주입
- canonical node code로 static·OD·거리·인접·mask·code를 같은 순서로 reindex
- Dataset과 같은 feature 이름 정렬과 학습 scaler 결과를 재현
- 선택 신도시 mask를 생성하고 OD 행·열을 0 처리
- 도시별 데이터와 DB `ModelVersion`/`ModelNode`를 같은 node code·순서로 연결
- v7 고정 node smoke를 데이터 묶음의 `x_static.shape[0]` 기준 검증으로 교체
- 최종 Adapter 의존성을 백엔드 실행 환경에 추가
- model·input-data version 변경 시 `PREDICTION_CACHE_VERSION`을 갱신

현재 백엔드의 동일 요청 cache와 성공·실패·cache hit 기록은 `/analyze/`
경로에 적용된다. `/predict/`에는 적용되지 않는다.

## AI/백엔드 역할 분담

| 담당 | 항목 |
|---|---|
| AI 팀 | adapter에 포함된 모델 코드·checkpoint, feature schema, canonical node·zone code와 순서, OD·거리·인접 데이터 또는 생성 규칙, mask 규칙, scaler 재현 데이터·분할·코드 또는 저장 parameter, model/input-data version·checksum |
| 백엔드 | 요청 code에 맞는 데이터 선택, settings, concrete preprocessor, Provider wrapper, DB model/node, cache·실행 기록·Matrix·지도·인사이트 |

경계·가상 node와 인구·static feature 배분은 경훈님이 작업 중이다. 도시별 기본
연령 비율은 연령 비율 예측 모델 결과로 추후 제공할 예정이다. 단계별 인구와
기타 기본값은 AI 팀에서 아직 확정하지 않았다.

## 모델 버전 매핑

Adapter raw `metadata.model_version`은 checkpoint 파일명의 stem이다. 이 값은 Django
`ModelVersion.code`와 다를 수 있다.

현재 `normalize_prediction_result()`는 Adapter metadata를 그대로 보존하고,
`region_context`는 그 `model_version`으로 DB `ModelNode`를 조회한다. 따라서 checkpoint
stem과 DB `ModelVersion.code`가 다르면 지도·Matrix용 node 조회가 비어 있을 수 있다.

모델 버전 정규화는 현재 구현된 동작이 아니라 mae-year 연결 시 백엔드가 반드시
구현해야 하는 항목이다. `MAE_MODEL_VERSION_CODE` 같은 설정을 추가할지 활성 DB
`ModelVersion`을 조회할지는 백엔드 팀이 결정하고, 선택한 code로 metadata와
`ModelNode` 조회를 일치시켜야 한다.

## 운영 데이터 연결 전 제한

`MAEProvider.predict()`는 concrete `PopulationPreprocessor`가 없거나 필수 도시 데이터를
찾을 수 없으면 `PreprocessingConfigurationError`를 발생시킨다. 운영 데이터 연결
전에는 합성값이나 프론트 더미 값으로 고수준 `predict()`를 실행하지 않는다.
