# OD 데이터 전처리 및 모델 코드

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
    * 코드 정리 중

## 모델 학습

작성중

## 모델 구조

작성중

## baseline 성능

작성중
