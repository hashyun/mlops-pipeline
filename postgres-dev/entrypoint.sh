#!/bin/bash
set -e

# 1. airflow_db에 airflow 유저 권한 부여
psql -U postgres -d airflow_db -c "GRANT ALL ON SCHEMA public TO airflow;"

# 2. mlflow_db에 mlflow 유저 권한 부여
psql -U postgres -d mlflow_db -c "GRANT ALL ON SCHEMA public TO mlflow;"
