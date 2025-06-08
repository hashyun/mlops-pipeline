# 👋&nbsp; Hi there!

고려대 통계학과 데이터애널리틱스특수연구3 프로젝트~

## Requirements

- python >= 3.10
- Airflow = 2.10.5

## Usage
파이프라인을 실행하려면 <code>src/config</code> 디렉터리의 숨김 파일 <code>.env.example</code>을 <code>.env</code>로 복사해야 합니다. 이 과정을 포함해 컨테이너 실행까지 자동으로 수행하려면 다음 스크립트를 실행하세요.

```bash
./run_pipeline.sh
```

스크립트는 <code>.env</code> 파일이 없을 경우 예제 파일을 복사한 뒤 Docker Compose를 이용해 필요한 컨테이너들을 띄웁니다.

텍스트 정제나 감성 분석 기능을 사용하려면 별도로 <code>nltk</code> 등의 패키지를 설치해야 합니다. Airflow에서 LSTM 파이프라인만 실행할 경우 이 의존성들이 없어도 됩니다.



### LSTM Pipeline
새롭게 추가된 DAG `lstm_pipeline`은 `merged_train (1).csv`와 `merged_test (1).csv` 데이터를 사용해 LSTM 모델을 학습합니다. 모델 학습과 예측 파일 생성은 Airflow에서 자동으로 실행됩니다.

학습이 끝나면 `notebook/weights/lstm_model.pth`에 모델 가중치가 저장되고, 예측 결과는 `notebook/data/predictions.csv`로 출력됩니다.

### Dash App
`dash_app/app.py`를 실행하면 위에서 생성된 예측 결과를 시각화하는 대시보드를 확인할 수 있습니다.

```bash
python dash_app/app.py
```
