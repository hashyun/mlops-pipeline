
### docker build --force-rm -t [image name]:[tag] -f [dockerfile filename] .

# 1. 베이스 이미지 설정
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /home/work
RUN mkdir /home/work/notebook

# 3. 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# 4. Python 패키지 설치
RUN pip install --no-cache-dir jupyterlab notebook psycopg2-binary sqlalchemy pandas numpy scikit-learn lightgbm xgboost 

# 5. JupyterLab 기본 설정 (포트와 설정 파일)
EXPOSE 8888
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# 6. 실행 명령 설정
CMD ["jupyter-lab", "--allow-root", "--config=/root/.jupyter/jupyter_notebook_config.py"]