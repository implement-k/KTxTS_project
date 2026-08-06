# OD 데이터 전처리 및 모델 코드

## 백엔드 연동

최종 `src/mae-year/` 모델의 Django Provider, 입력 DTO, 출력 Adapter, 호출 예시와 실제
checkpoint smoke test는 [`mae_backend_adapter/README.md`](mae_backend_adapter/README.md)에
정리되어 있다. 기본 checkpoint는 `best_model/mae:hybrid-86epoch.pth`이며 모델 학습 구조는
연동 코드에서 수정하지 않는다.

❗ 주의 ❗ 모델 수정할때는 각 브랜치에서 수정

## 파일 구조

* KTDB
  * 📁 dataset : 데이터셋 처리관련 코드 및 데이터셋 파일
    * 📁 preprocessing : raw 데이터 가공 코드
      * make_*: 모델 input 값 생성
      * process_*: make_static_feature에서 쓰이는 함수 파일
    * 📁 processed: KTDB/dataset/preprocessing/process_*에서 전처리한 데이터
    * 📁 raw: 디코에 보내준 파일 원본
    * final_static_feature_{year}.csv: 모델 input static features matrix[year: '2023' | '2019']
    * dist_data_{year}.csv: 행정동간 거리 matrix[year: '2023' | '2019']
    * od_data_{year}.csv: od데이터[year: '2023' | '2019']
    * od_static_feature_{year}.csv: 비수도권 이동 데이터[사용 안함.][year: '2023' | '2019']
  * 📁 src: 모델 코드
    * 📁 mae-year: SSL방식의 모델
      * ❗models.py: 모델 코드[이 코드 수정하면 됨]
      * ❗train.py: 학습 코드[이 코드 수정하면 됨]
      * dataset.py: 데이터셋 로드 코드(마스킹, 테스트 데이터셋 분리 등)
      * loss.py: 모델에서 쓸 loss 함수들
    * 📁 gravity(경훈): 기존 회귀모델+중력모델
    * 📁 deepgravity: 기존 회귀모델+fnn 모델. (두 단계 모두 각 브랜치에서 자유롭게 수정)
      * ❗models.py: 모델 코드[이 코드 수정하면 됨]
      * ❗train.py: 학습 코드[이 코드 수정하면 됨]
    * loss.py: 모델에서 쓸 loss 함수들
    * main.py: 이전에 테스트했던 모델(argument없이 실행)
    * validation.py: 모델 validation코드
    * config.py: 모델 하이퍼파라미터, 파일 path 설정 파일

## dataset 사용법

raw 폴더에 기존 디코에 보낸 형식으로 올리고, make파일 실행
process 파일에서 행정동 합치는 코드까지 있으니까 raw파일 형식대로만 넣고 돌리면 알아서 됩니다.
** 주의: 파일 path 자신의 컴퓨터에 맞게 수정 **

#### 사용법

1. google drive에 kt폴더 생성
2. KTDB폴더 kt폴더 안에 넣기
3. colab에 colab.ipynb올리고 1번셀 실행
4. 구글 드라이브 연동 완료 후 실행하고 싶은 모델 실행(epoch, batch 조절. 배치크기 64 넘어가면 OOM발생하므로 64미만 추천)

## 중력모델 성능

중력모델은 LGBM으로 예측한 행정동별 외부유출·외부유입 총량을 기준으로,  가까운 행정동끼리는 더 많이 연결되고, 먼 행정동끼리는 적게 연결되도록 거리 효과를 반영했다.
테스트 지역(동탄·위례·검단 포함 OD) 기준 성능은 다음과 같음 (beta :2.0 거리 가중치)

- RMSE: 521.000
- CPC: 0.5608
- IPF는 10회 반복에서 수렴했으며, 행합 상대오차는 0.000067, 열합 상대오차는 0.000000으로 예측 총량 제약을 안정적으로 만족하였음.
