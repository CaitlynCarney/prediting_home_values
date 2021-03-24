import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, RFE, SelectKBest
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 

from env import host, user, password

from math import sqrt
import seaborn as sns
import warnings
from pydataset import data



def select_kbest(x, y, k):
    '''
    Parameters:
    x = dataframe
    y = target,
    k = # features to return
    '''
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(x, y)
    object_cols = get_object_cols(x)
    numeric_cols = get_numeric_X_cols(X_train, object_cols)
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X_train_scaled, y_train)
    feature_mask = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    
    return f_feature
