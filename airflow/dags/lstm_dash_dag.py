from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from airflow_utils import lstm_train_task, prediction_task

DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    dag_id='lstm_pipeline',
    default_args=DEFAULT_ARGS,
    schedule_interval='@daily',
    catchup=False
)

with dag:
    train = PythonOperator(
        task_id='train_lstm',
        python_callable=lstm_train_task
    )

    predict = PythonOperator(
        task_id='generate_predictions',
        python_callable=prediction_task
    )

    train >> predict
