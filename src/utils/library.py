from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


class numeric_filtering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.constant_col = [i for i in range(X.shape[1]) if X[:,i].std()==0]
        self.id_col = [i for i in range(X.shape[1]) if len(np.unique(np.diff(X[:,i])))==1]
        self.rm_cols = self.constant_col + self.id_col
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self
    
    def transform(self, X):
        result = X[:,self.final_cols]
        return result
    
    
class categorical_filtering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.constant_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i]))==1]
        self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i]))==X.shape[0]]
        self.cardinality = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) > 50]
        self.rm_cols = self.constant_col + self.id_col + self.cardinality
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self
    
    def transform(self, X):
        result = X[:,self.final_cols]
        return result
    
def load_data(data_path = None):
    from catboost.datasets import titanic

    train, test = titanic()
    ycol = 'Survived'
    xcol = [col for col in train.columns if col not in [ycol]]

    return train, test, xcol, ycol

def data_preparation(data_path, table_name):
    import pandas as pd
    import numpy as np
    from psycopg2 import connect, extensions
    from sqlalchemy import create_engine

    user = 'admin'
    pw = 'admin'
    ip = '172.19.0.2'
    dbname = 'postgres'
    conn_string = f'postgresql://{user}:{pw}@{ip}/{dbname}'
    db = create_engine(conn_string)

    try:
        conn = db.connect()
        dat = pd.read_csv(data_path)
        dat.to_sql(name=table_name, con=conn, if_exists='replace', index=False) 
        conn.commit()
        print('Success!!')
    except Exception as e:
        print(e)
    finally:
        conn.close()


def data_preparation_from_db(table_name):
    import pandas as pd
    from psycopg2 import connect, extensions

    try:
        conn = connect(database="postgres",
                host='172.19.0.1',
                port=5433,
                user='admin',
                password='admin')
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        
        df = pd.DataFrame(data=cursor.fetchall(),
                        columns=[x.name for x in cursor.description])
        
        cursor.close()
        conn.close()
        print('Success!!')

        return df
    except Exception as e:
        print(e)
    finally:
        conn.close()
