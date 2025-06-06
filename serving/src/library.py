import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier



class Nonvalue_transform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        result = X.replace({None: np.nan})
        return result


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


class ensemble_model:
    def __init__(self):
        
        self.models = {
            'RF': RandomForestClassifier(),
            'XGB' : XGBClassifier(),
            'LGBM' : LGBMClassifier(),
        }
        
        pipe1 = Pipeline([
            ('step1',   SimpleImputer(strategy="mean") ),
            ('step2',   numeric_filtering()  ),
            ('step3',   StandardScaler()  ),
        ]) 
        
        pipe2 = Pipeline([
            ('step1',   SimpleImputer(strategy="most_frequent") ),
            ('step2',   categorical_filtering()  ),
            ('step3',   OneHotEncoder()  ),
        ])

        self.transform1 = Nonvalue_transform()
        self.transform2 = ColumnTransformer([
            ('num',  pipe1,  make_column_selector(dtype_include=np.number)),
            ('cat',  pipe2,  make_column_selector(dtype_exclude=np.number)),
        ])

    
    def fit(self, X, y):
        self.columns = X.columns.tolist()
        
        ensemble_models = []
        for model in self.models.keys():
            pipe0 = Pipeline([
                ('trainform1', self.transform1),
                ('transform2', self.transform2),
                ('model', self.models[model])
            ])
            ensemble_models.append( (model, pipe0) )
        
        self.ensemble = VotingClassifier(ensemble_models, voting="soft", verbose=0)
        self.ensemble.fit(X,y)    
        self.imp = permutation_importance(estimator = self.ensemble,
                                          X = X, y = y, scoring="accuracy", n_repeats=5 )
        return self

    
    def predict(self, X):
        pred = pd.DataFrame(self.ensemble.predict_proba(X)[:,0], columns=['ensemble'])
        for model in self.models.keys():
            pred[model] = self.ensemble.named_estimators_[model].predict_proba(X)[:,0]
        return pred
        
    def feature_importances(self):
        result = pd.DataFrame(self.imp['importances_mean'], index=self.columns, columns=['features'])
        result = result.sort_values('features')
        return result


