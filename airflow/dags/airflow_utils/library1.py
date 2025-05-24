import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance

class numeric_filtering(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.constant_col = [i for i in range(X.shape[1]) if X[:,i].std()==0]
        self.id_col = [i for i in range(X.shape[1]) if len(np.unique(np.diff(X[:,i])))==1]
        self.rm_cols = self.constant_col + self.id_col
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        result = X[:,self.final_cols]
        return result

class categorical_filtering(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.constant_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i]))==1]
        self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i]))==X.shape[0]]
        self.cardinality = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) > 50]
        self.rm_cols = self.constant_col + self.id_col + self.cardinality
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        result = X[:,self.final_cols]
        return result


def load_data():
    from catboost.datasets import titanic

    train, test = titanic()
    ycol = 'Survived'
    xcol = [col for col in train.columns if col not in [ycol]]

    return train, test, xcol, ycol

def train_model(**context):
    train, test, xcol, ycol = context['task_instance'].xcom_pull(task_ids='load_data')
    
    # 파이프라인 구성
    pipe1 = Pipeline([
        ('step1', SimpleImputer(strategy="mean")),
        ('step2', numeric_filtering()),
        ('step3', StandardScaler()),
    ])
    
    pipe2 = Pipeline([
        ('step1', SimpleImputer(strategy="most_frequent")),
        ('step2', categorical_filtering()),
        ('step3', OneHotEncoder()),
    ])
    
    transform = ColumnTransformer([
        ('num', pipe1, make_column_selector(dtype_include=np.number)),
        ('cat', pipe2, make_column_selector(dtype_exclude=np.number)),
    ])
    
    pipe = Pipeline([
        ('transform', transform),
        ('model', RandomForestClassifier())
    ])
    
    pipe.fit(train[xcol], train[ycol])
    
    imp = permutation_importance(
        estimator = pipe,
        X = train[xcol],
        y = train[ycol],
        scoring = "accuracy",
        n_repeats = 5
    )
    
    feature_importance = pd.DataFrame(
        imp['importances_mean'],
        index = xcol,
        columns = ['features']
    )
    
    predictions = pipe.predict(test[xcol])
    
    context['task_instance'].xcom_push(key = 'predictions', value = predictions.tolist())
    context['task_instance'].xcom_push(key = 'feature_importance', value = feature_importance.to_dict())


def data_preprocessing(**kwargs):
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    ti = kwargs['ti']
    ti.xcom_push(key = 'X_train', value = X_train.to_json())
    ti.xcom_push(key = 'X_test', value = X_test.to_json())
    ti.xcom_push(key = 'y_train', value = y_train.to_json(orient = 'records'))
    ti.xcom_push(key = 'y_test', value = y_test.to_json(orient = 'records'))

def train_model(model_name, **kwargs):
    ti = kwargs['ti']
    X_train = pd.read_json(ti.xcom_pull(key = 'X_train', task_ids = 'data_preprocessing'))
    X_test = pd.read_json(ti.xcom_pull(key = 'X_test', task_ids = 'data_preprocessing'))
    y_train = pd.read_json(ti.xcom_pull(key = 'y_train', task_ids = 'data_preprocessing'), typ = 'series')
    y_test = pd.read_json(ti.xcom_pull(key = 'y_test', task_ids = 'data_preprocessing'), typ = 'series')

    if model_name == 'RandomForest':
        model = RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Unsupported model: " + model_name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    performance = accuracy_score(y_test, predictions)

    ti.xcom_push(key=f'performance_{model_name}', value=performance)

def select_best_model(**kwargs):
    ti = kwargs['ti']
    rf_performance = ti.xcom_pull(key = 'performance_RandomForest', task_ids = 'train_rf')
    gb_performance = ti.xcom_pull(key = 'performance_GradientBoosting', task_ids = 'train_gb')

    best_model = 'RandomForest' if rf_performance > gb_performance else 'GradientBoosting'
    print(f"Best model is {best_model} with performance {max(rf_performance, gb_performance)}")

    return best_model

