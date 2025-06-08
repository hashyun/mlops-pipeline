from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add repository root to sys.path so the DAG can import src.lstm_pipeline
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.lstm_pipeline import run_lstm_pipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='lstm_pipeline',
    default_args=default_args,
    description='Train LSTM model and generate predictions',
    schedule_interval='@daily',
    catchup=False,
)

with dag:
    run_lstm = PythonOperator(
        task_id='run_lstm_pipeline',
        python_callable=run_lstm_pipeline,
    )
