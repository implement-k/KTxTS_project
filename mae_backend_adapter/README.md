# mae-year 백엔드 연결 안내

## 1. 한눈에 보기

처음 읽을 때는 아래 순서로 보면 된다.

1. 이 README에서 파일 위치와 연결 흐름을 확인한다.
2. Django에 붙일 때는 [`docs/backend_integration.md`](docs/backend_integration.md)를 본다.
3. 모델 입력을 만들 때는 [`docs/model_contract.md`](docs/model_contract.md)를 본다.

이 패키지는 mae-year 모델을 Django에서 호출하기 위한 연결 코드다.

- 모델은 서버 process(독립 실행 단위)마다 한 번 불러오고 요청마다 다시 사용한다.
- node(모델이 계산하는 지역 단위) 수 `N`은 고정값이 아니다. 입력 데이터 크기로 정한다.
- 요청에서 `N`을 직접 받지 않는다.
- 결과는 신도시와 관련된 일평균 OD 목록이다. OD는 출발지와 도착지 사이의 이동량이다.

## 2. 백엔드가 볼 파일

경로는 저장소 최상위 폴더를 기준으로 한다.

| 경로 | 역할 |
|---|---|
| [`mae_backend_adapter/`](./) | Django가 모델을 호출할 때 사용하는 Python 패키지 |
| [`mae_backend_adapter/predictor.py`](predictor.py) | 요청을 확인하고 모델 실행 결과를 OD 목록으로 바꾸는 코드 |
| [`mae_backend_adapter/static_scaler.py`](static_scaler.py) | 지역별 고정 특성값을 학습 때와 같은 기준으로 변환하는 코드 |
| [`mae_backend_adapter/artifacts/`](artifacts/) | 2023년 특성값 변환 기준을 담은 NPZ·JSON 파일 |
| [`mae_backend_adapter/requirements.txt`](requirements.txt) | 실행에 필요한 Python 패키지 목록 |
| [`mae_backend_adapter/model/mae.py`](model/mae.py) | 배포할 `ODMAE` 모델 구조 |
| [`mae_backend_adapter/model/mae.pth`](model/mae.pth) | 배포할 학습 가중치 |
| 학습 저장소 `dataset/newtown.zip` | 도시별 신도시 입력 데이터. 백엔드 전처리 연결 시 별도 배포 |

활성화된 Python 환경에 의존성을 설치한다.

```bash
python -m pip install -r mae_backend_adapter/requirements.txt
```

보통 백엔드 연결 작업은 `predictor.py`와 `docs/backend_integration.md`부터 보면 된다.
입력 배열의 순서와 크기를 다룰 때만 `docs/model_contract.md`까지 확인한다.

## 3. 요청과 결과

공개 Django API가 받는 값은 다음 세 가지다.

| 요청값 | 설명 |
|---|---|
| `newtown_code` | 신도시를 찾기 위한 코드. 백엔드가 이 코드로 도시 이름을 조회한다. |
| `total_population` | 적용할 전체 인구. 공개 API에서는 1 이상의 정수다. |
| `age_ratios` | `0_19`, `20_59`, `60_plus`의 비율. 세 값의 합은 1이어야 한다. |

백엔드는 `newtown_code`와 인구·연령 비율을 `MAEProvider.predict()`에
전달한다. 현재 지원 code는 `gyosan`, `wangsuk`, `changneung`, `all`이다.

`MAEProvider`의 주요 반환값은 다음과 같다.

| 반환값 | 설명 |
|---|---|
| `newtown_zone_codes` | 선택한 신도시에 속한 node 코드 목록 |
| `od` | 신도시 내부 이동, 유입, 유출을 담은 일평균 이동량 목록 |
| `metadata` | 모델 버전, node 수 등 실행 정보를 담은 값 |

`od`의 각 항목에는 `origin_code`, `destination_code`, `predicted_trips`와
`movement_type`이 있다. `movement_type`은 `internal`, `outflow`, `inflow` 중 하나다.

전체 OD 행렬과 시간대별 예측은 반환하지 않는다.
Django의 최종 API 응답 형식은 백엔드 서비스 계층에서 따로 만든다.

## 4. 백엔드 연결 방법

각 Django worker는 `MAEProvider`를 한 번 만들고 같은 객체를 계속 사용해야 한다.
요청마다 새로 만들면 모델 파일을 매번 다시 읽게 된다.

핵심 호출은 아래와 같다. `get_mae_provider()`는 같은 process 안에서 같은 객체를
돌려주도록 백엔드에서 구현한다.

```python
provider = get_mae_provider()
result = provider.predict(
    newtown=request.newtown_code,
    total_population=request.total_population,
    age_ratios={key: float(value) for key, value in request.age_ratios.items()},
)
```

`PopulationPreprocessor`는 백엔드가 구현할 “도시별 모델 입력 생성기”의
인터페이스 이름이다.

현재 패키지는 이 구현부가 따라야 할 규칙만 제공한다.
도시별 데이터를 고르고 모델 입력으로 바꾸는 코드는 백엔드에서 추가로 구현해야 한다.
프론트의 임시 값을 운영 기본값으로 사용하면 안 된다.

권장 구현 위치, Provider 선택 설정, 모델 경로 설정, 버전 연결 방법은
[`docs/backend_integration.md`](docs/backend_integration.md)에 정리되어 있다.
Django 세부 변경사항은 해당 문서를 기준으로 작업한다.

## 5. 아직 필요한 데이터

### 실제 도시 모델 입력을 만들기 위해 필요한 데이터

- 도시별 가상 node의 최종 코드와 순서
- 가상 node별 static feature 배분 방법
- 요청으로 받은 인구·연령 비율을 가상 node에 배분하는 방법
- 가상 node가 포함된 OD·거리·인접 행렬 데이터 또는 생성 규칙

이 데이터와 도시별 입력 생성 코드가 없으면 실제 교산·창릉·왕숙 예측을 실행할 수 없다.

### 추후 정할 화면·시나리오 기본값

- 초기·중기·완료 단계 인구
- 도시별 기본 연령 비율

두 값은 화면의 기본 시나리오를 위한 값이다. 사용자가 `total_population`과
`age_ratios`를 직접 전달하는 모델 호출에는 필수 데이터가 아니다.

## 6. 상세 문서

- 모델 입력 크기, node 순서, 특성값 변환: [`docs/model_contract.md`](docs/model_contract.md)
- 현재 Django 백엔드 연결 항목과 설정: [`docs/backend_integration.md`](docs/backend_integration.md)

테스트는 저장소 최상위 폴더에서 실행한다.

```bash
python -m unittest discover -s mae_backend_adapter/tests -v
```

이 테스트는 연결 코드의 요청 검사, 입력 크기 검사, 모델 파일 로드와 결과 변환을
확인한다. 아직 필요한 도시별 운영 데이터가 준비됐음을 뜻하지는 않는다.
