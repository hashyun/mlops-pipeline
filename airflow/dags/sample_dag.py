from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable

from datetime import datetime, timedelta

from airflow_utils import data_preprocessing, train_model, select_best_model

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
    dag_id = 'sample_feature_extraction',
    default_args = default_args,
    description = 'A simple DAG for feature extraction',
    schedule_interval = '@daily',
    catchup = False
)

with dag:
    t1 = PythonOperator(
        task_id = 'data_preprocessing',
        python_callable = data_preprocessing,
    )

    t2 = PythonOperator(
        task_id = 'train_rf',
        python_callable = train_model,
        op_kwargs = {'model_name': 'RandomForest'},
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id = 'train_gb',
        python_callable = train_model,
        op_kwargs = {'model_name': 'GradientBoosting'},
        provide_context=True,
    )

    t4 = PythonOperator(
        task_id = 'select_best_model',
        python_callable = select_best_model,
        provide_context = True,
    )

    t1 >> [t2, t3] >> t4
