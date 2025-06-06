### docker exec -d [container_id] uvicorn [.py filename]:app --host 0.0.0.0 --port 8080 

# 1. 베이스 이미지 설정
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /home/app

# 3. 필요한 시스템 패키지 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 패키지 설치
RUN pip install --no-cache-dir fastapi uvicorn pydantic psycopg2-binary sqlalchemy numpy pandas scikit-learn lightgbm xgboost 

# 5. FastAPI 기본 설정 (포트와 설정 파일)
EXPOSE 8089

# CMD ["uvicorn", "main:app", "--reload", "--app-dir", "/home/app/"]
CMD ["uvicorn", "main:app", "--reload", "--host=0.0.0.0", "--port=8089", "--app-dir", "/home/app/"]
