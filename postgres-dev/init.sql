-- airflow 전용 유저 및 DB
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow_db OWNER airflow;

-- mlflow 전용 유저 및 DB
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow_db OWNER mlflow;
