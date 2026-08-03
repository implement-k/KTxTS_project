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
    newtown=request.newtown_name,
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
`mae_loader.py`는 model·LightGBM·preprocessor를 Django worker process당 한 번 로딩한다.
`mae_preprocessor.py`는 선택된 도시별 모델 입력 데이터 묶음을 `ModelInputs`로 바꾼다.

## 필요한 Django settings

| 설정 | 의미 |
|---|---|
| `MAE_HANDOFF_ROOT` | Adapter와 모델 소스의 배포 root |
| `MAE_MODEL_PATH` | 최종 MAE checkpoint |
| `MAE_LGBM_MODEL_PATH` | LightGBM self-loop 모델 |
| `MAE_MODEL_MODULE_PATH` | `src/mae-year/models.py` |
| `MAE_PREPROCESSING_ARTIFACT_ROOT` | 도시별 모델 입력 데이터 root |
| `MAE_ARTIFACT_MANIFEST_PATH` | node·feature·scaler·version manifest |
| `MAE_MODEL_VERSION_CODE` | Django에서 사용할 안정적 deployment version |
| `MAE_DEVICE` | `cpu`, `cuda` 등 PyTorch device |

경로와 version 값은 배포 데이터가 확정된 뒤 설정하며 Adapter에 기본값을
임의로 하드코딩하지 않는다.

## 백엔드가 수정할 항목

- 기존 v7 model source 경로를 `src/mae-year/models.py`로 변경
- mae-year Provider 선택값을 Provider factory에 추가
- `TensorShapeError`를 `InputValidationError`보다 먼저 예외 변환
- concrete `PopulationPreprocessor`를 loader에 주입
- canonical node code로 static·OD·거리·인접·mask·code를 같은 순서로 reindex
- Dataset과 같은 feature 이름 정렬과 학습 scaler 결과를 재현
- 현재 운영에서 전체 active mask와 선택 신도시 mask를 생성하고 OD 행·열을 0 처리
- 도시별 데이터와 DB `ModelVersion`/`ModelNode`를 같은 node code·순서로 연결
- v7 고정 node smoke를 데이터 묶음의 `x_static.shape[0]` 기준 검증으로 교체
- 최종 Adapter 의존성을 백엔드 실행 환경에 추가
- model·input-data version 변경 시 cache version을 갱신

현재 백엔드의 동일 요청 cache와 성공·실패·cache hit 기록은 `/analyze/`
경로에 적용된다. `/predict/`에는 적용되지 않는다.

## AI/백엔드 역할 분담

| 담당 | 항목 |
|---|---|
| AI 팀 | 모델 코드·checkpoint·LightGBM, feature schema, canonical node·zone code와 순서, OD·거리·인접 데이터 또는 생성 규칙, mask 규칙, scaler 재현 데이터·분할·코드 또는 저장 parameter, model/input-data version·checksum |
| 백엔드 | 요청 code에 맞는 데이터 선택, settings, concrete preprocessor, Provider wrapper, DB model/node, cache·실행 기록·Matrix·지도·인사이트 |

경계·가상 node와 인구·static feature 배분은 경훈님이 작업 중이다. 도시별 기본
연령 비율은 연령 비율 예측 모델 결과로 추후 제공할 예정이다. 단계별 인구와
기타 기본값은 AI 팀에서 아직 확정하지 않았다.

## 모델 버전 매핑

Adapter raw `metadata.model_version`은 checkpoint 파일명의 stem이다. 이 값은 Django
`ModelVersion.code`와 다를 수 있다.

백엔드는 `MAE_MODEL_VERSION_CODE`에 안정적인 deployment version을 설정하고 Adapter
응답의 `metadata.model_version`을 이 값으로 정규화한다. 같은 code를 DB `ModelVersion`과
`ModelNode` 조회에 사용한다.

## 운영 데이터 연결 전 제한

`MAEProvider.predict()`는 concrete `PopulationPreprocessor`가 없거나 필수 도시 데이터를
찾을 수 없으면 `PreprocessingConfigurationError`를 발생시킨다. 운영 데이터 연결
전에는 합성값이나 프론트 더미 값으로 고수준 `predict()`를 실행하지 않는다.
