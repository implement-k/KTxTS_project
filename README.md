# OD 데이터 전처리 및 모델 코드

❗ 주의 ❗ 모델 수정할때는 각 브랜치에서 수정

## 파일 구조

* KTDB
  * 📁 dataset : 데이터셋 처리관련 코드 및 데이터셋 파일
    * 📁 preprocessing : raw 데이터 가공 코드
      * make_*: 모델 input 값 생성
      * process_*: make_static_feature에서 쓰이는 함수 파일
    * 📁 processed: KTDB/dataset/preprocessing/process_*에서 전처리한 데이터
    * 📁 raw: 디코에 보내준 파일 원본
    * final_static_feature.csv: 모델 input static features matrix
    * dist_data.csv: 행정동간 거리 matrix
    * od_data.csv: od데이터
    * od_static_feature.csv: 비수도권 이동 데이터[현재 사용 안함.]
  * 📁 src: 모델 코드
    * 📁 model: 모델 학습, 테스트, 데이터셋 코드
      * colab.ipynb: colab용 코드(사용법은 아래 설명 참조)
      * dataset.py: 데이터셋 로드 코드(마스킹, 테스트 데이터셋 분리 등)
      * ❗models.py: 모델 코드[이 코드 수정하면 됨]
      * ❗train.py: 학습 코드[이 코드 수정하면 됨]
    * 📁 gravity(경훈): 기존 회귀모델+중력모델
    * 📁 twostage: 기존 회귀모델+fnn 모델. (두 단계 모두 각 브랜치에서 자유롭게 수정)
    * config.py: 모델 하이퍼파라미터, 파일 path 설정 파일

## dataset 사용법

raw 폴더에 기존 디코에 보낸 형식으로 올리고, make파일 실행
process 파일에서 행정동 합치는 코드까지 있으니까 raw파일 형식대로만 넣고 돌리면 알아서 됩니다.
** 주의: 파일 path 자신의 컴퓨터에 맞게 수정 **

## 모델 학습(colab.ipynb)

colab T4 사용시 한 에포크 당 2분 소요.(KT에서 지원해주므로 pro 결제하고 H100 사용하면 더 빠르게 학습 될 듯)

우리는 먼저 mae1모델을 우선적으로 학습시켜야함.

#### 사용법

1. google drive에 kt폴더 생성
2. KTDB폴더 kt폴더 안에 넣기
3. colab에 colab.ipynb올리고 1번셀 실행
4. 구글 드라이브 연동 완료 후 실행하고 싶은 모델 실행(epoch, batch 조절. 배치크기 64 넘어가면 OOM발생하므로 64미만 추천)

#### 학습법

1. mae모델은 랜덤으로 행정동 선택하고 그 주위까지 포함하여 k개의 행정동 선택함.
2. 선택된 k개의 행정동의 OD, static feature의 종사자수, 사업체수를 0으로 masking하고, 마지막 열(masking유무)을 1로 바꿈.
3. loss 계산 (heavy-tail MSE)
   1. 현재 log1p를 적용한 상태이므로 큰 통행량을 잘 못 맞추는 경향이 있어서 추가함.

## 모델 구조

코드 참조

## baseline 성능

mae1의 경우 정답률 54%
