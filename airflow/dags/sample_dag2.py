from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from airflow_utils import train_model, load_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes = 5),
}

dag = DAG(
    dag_id = 'titanic_pipeline',
    default_args = default_args,
    description = '타이타닉 파이프라인',
    schedule_interval = timedelta(days = 1),
    catchup = False
)

with dag:
    t1 = PythonOperator(
        task_id = 'load_data',
        python_callable = load_data,
        dag = dag
    )

    t2 = PythonOperator(
        task_id = 'train_model',
        python_callable = train_model,
        provide_context = True,
        dag = dag
    )

    t1 >> t2 
