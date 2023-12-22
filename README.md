# Mask Image Classification

### 대회결과


- 최종순위 : 5등
- F1 score : 0.7362
- acc: 79.5079

### 맡은 역할

- 은성 : 데이터 전처리, 증강 실험
- 유민 : EDA를 통한 데이터 분석 / 데이터 라벨 수정 실험
- 현영 : 멀티 레이블 코드 개발
- 태민 : 새롭게 구성된 전체 Base 코드 배포 / 모델 학습 실험
- 종진 : 앙상블 실험 및 적용

### 기술적 시도

- 데이터 불균형 처리 → Oversampling, DownSampling
- 데이터 전처리 시도 → 이미지 Background 제거, Crop

### 모델

- Efficientnetb0 b3 , Swin-t , Resnet

 

- 각자 best 모델의 결과를 바탕으로 앙상블 - Soft Voting

- 멀티 레이블 구조 모델과 싱글 레이블 모델 비교 실험
    - 하나의 레이블로 학습 했을때 더 높은 성능
- Fine - tunning : 모델 전체 다 학습 실험과 앞부분을 freeze하고 뒷단만 학습 비교
    - 모델 전체를 학습하는 경우가 더 높은 성능

### Loss 함수

- cross entropy
- label smoothing
    - 개별 최고 점수 모델에서 활용
- f1 Loss
    - 실험 결과 성능이 떨어짐을 확인

### Optimizer

- SGD
- Adam
- AdamW - fix

### 데이터

- RemoveBG를 사용한 배경제거를 한 데이터셋을 사용
    - 일부 데이터가 배경의 일부 잘못 제거됨 → 오히려 성능 하락을 보였음
    
- 데이터 증강
    - BaseAugmentation
    - Sharpen, Contrast
    
- 연령대의 불균형 해결을 위한 Sampling
    - OverSampling : 30 ~ 59 →3 배 / 60대 → 7배
    - DownSampling : 56 ~ 59 → Drop

![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-05/assets/98599867/6c8edfe7-b9e2-4a15-b1b5-d186a57c6de1)


- 성별 불균형

![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-05/assets/98599867/dfcbafcb-beca-41d6-8924-1a44a064e3cc)


## 배운점


- 저희는 당장의 성과보다는 코드 및 구조를 이해하고 재구성 하는법을 배우고 효과적인 개발 및 협력 환경을 구성하는 것도 중요하다는 것을 배웠습니다.
- EDA의 중요성
- 데이터 imbalance 문제해결에 대한 다양한 방법 고안

### 아쉬운 점

- 실험을 체계적으로 기록하지 못함
- Git을 제대로 활용하지 못함
- acc → f1 score 제대로 된 평가 Matrix 구성하지 못함
- pre-trained 모델로 멀티레이블를 사용하려고 했을 시 forward를 재정의 해야 함

- class imbalance를 해결하기위한 loss function 변경 : f1 loss
- 과적합 발생했을 때 모델 크기를 줄여보는 시도

### 공부할것
- focal loss
- class imabalance 해결방법
- SGD 와 AdamW
- label smoothing : 레이블의 노이즈가 있을때 사용
