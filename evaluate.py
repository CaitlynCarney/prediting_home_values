import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from sklearn.preprocessing import MinMaxScaler

import prepare
import acquire
import explore
import warnings
warnings.filterwarnings('ignore')

def xtrain_xval_xtest():
    '''create X_train, X_validate, X_test, y_train, y_validate, y_test'''
    # pull from acquire and prepare
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # split the data
    train, validate, test = prepare.split_focused_zillow(df)
    X_train = train.drop(columns = ['appraised_value'])
    y_train = train.appraised_value
    X_validate = validate.drop(columns=['appraised_value'])
    y_validate = validate.appraised_value
    X_test = test.drop(columns=['appraised_value'])
    y_test = test.appraised_value
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def eval_y_train():
    '''Evaluate the y_train value
    determine whether to proceed with median or mean'''
    # grab the split up data
    X_train, y_train, X_validate, y_validate, X_test, y_test = xtrain_xval_xtest()
    # create basleine pedicted median
    home_value_baseline_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = round(home_value_baseline_median, 2)
    y_validate['appraised_value_pred_median'] = round(home_value_baseline_median, 2)
    # create baseline predicted mean
    home_value_baseline_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = round(home_value_baseline_mean, 2)
    y_validate['appraised_value_pred_mean'] = round(home_value_baseline_mean,2)
    return y_train, y_validate

def add_to_train():
    '''prepare train for the next steps'''
    # get the dataframe
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    train, validate, test = prepare.split_focused_zillow(df)
    # create the old model
    ols_model = ols('appraised_value ~ bedrooms', data=train).fit()
    # make new features
    train['yhat'] = round(ols_model.predict(train), 2)
    train['baseline'] = train.appraised_value.mean()
    train['residual'] = train.appraised_value - train.yhat
    train['baseline_residual'] = train.appraised_value - train.baseline
    train['residual_sqr'] = train.residual ** 2
    train['baseline_residual_sqr'] =  train.baseline_residual ** 2
    # run the SSE, MSE, and RMSE plus their baselines
    SSE = train['residual_sqr'].sum()
    SSE_baseline =  train['baseline_residual_sqr'].sum()
    MSE = SSE / len(df)
    MSE_baseline = SSE_baseline / len(df)
    RMSE = sqrt(MSE)
    RMSE_baseline = sqrt(MSE_baseline)
    return train

def SSE_MSE_RMSE():
    'Finds the Sum of Squares from add_to_train'
    # get the data
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    train, validate, test = prepare.split_focused_zillow(df)
    # pull from add to trian
    train = add_to_train()
    # set up for SSE
    SSE = train['residual_sqr'].sum()
    SSE_baseline =  train['baseline_residual_sqr'].sum()
    # set up for MSE
    MSE = SSE / len(df)
    MSE_baseline = SSE_baseline / len(df)
    # set up for RMSE
    RMSE = sqrt(MSE)
    RMSE_baseline = sqrt(MSE_baseline)
    return SSE, SSE_baseline, MSE, MSE_baseline, RMSE, RMSE_baseline

def SSE_MSE_RMSE_info():
    'Finds the SSE, MSE, and RMSE from add_to_train'
    # ge thte data
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = xtrain_xval_xtest()
    # make baseline predicted median 
    home_value_baseline_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = round(home_value_baseline_median, 2)
    y_validate['appraised_value_pred_median'] = round(home_value_baseline_median, 2)
    # make basleine predicted mean
    home_value_baseline_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = round(home_value_baseline_mean, 2)
    y_validate['appraised_value_pred_mean'] = round(home_value_baseline_mean,2)
    # set up for SSE
    train['residual_sqr'] = train.residual ** 2
    train['baseline_residual_sqr'] =  train.baseline_residual ** 2
    SSE = train['residual_sqr'].sum()
    SSE_baseline =  train['baseline_residual_sqr'].sum()
    print("SSE = ", round(SSE,3))
    print("SSE Baseline = ", round(SSE_baseline, 3))
    print('------------------------------------------')
    # set up for MSE
    MSE = SSE / len(df)
    MSE_baseline = SSE_baseline / len(df)
    print("MSE = ", round(MSE,3))
    print("MSE baseline = ", round(MSE_baseline,3))
    print('------------------------------------------')
    # set up for RMSE
    RMSE = sqrt(MSE)
    RMSE_baseline = sqrt(MSE_baseline)
    print("RMSE = ", round(RMSE,3))
    print("RMSE baseline = ", round(RMSE_baseline,3))
    print('------------------------------------------')
    # plot to visualize actual vs predicted. 
    sns.set(style="darkgrid")
    plt.hist(y_train.appraised_value, color='teal', alpha=.5, label="Actual Home Value")
    plt.vlines(y_train.appraised_value_pred_mean, 0, 5000, color='yellow', alpha=.3, label="Predicted Mean Home Value")
    plt.vlines(y_train.appraised_value_pred_median, 0, 5000, color='lawngreen', alpha=.2, label="Predicted Median Home Value")
    plt.xlabel("Home Value")
    plt.ylabel("Number Homes")
    plt.legend()
    plt.show()
    

def plot_residuals():
    ''' Gather tips data set from pydata
    add columns yhat, baseline, residualm and baseline residual to the df
    plot residual scatterplot
    plot baseline residual scatterplots'''
    # get the data
    tips = data("tips")
    # fit the model
    model = ols('total_bill ~ tip', data=tips).fit()
    # create new features
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
    # plot residuals
    plt.subplots(2, 1, figsize=(13,25), sharey=True)
    sns.set(style="darkgrid")
    plt.subplot(2,1,1)
    sns.scatterplot(x='tip',y='residual',data=tips, palette='rocket', hue='size')
    plt.axhline(y = 0, ls = ':', color='black', linewidth=4)
    plt.title('Model Residuals',fontsize = 20)
    plt.subplot(2,1,2)
    sns.scatterplot(x='tip',y='baseline_residual',data=tips,palette='rocket', hue='size')
    plt.axhline(y = 0, ls = ':', color='black', linewidth=4)
    plt.title('Baseline Residuals',fontsize = 20)
    

def regression_errors():
    ''' gather tips data set from pydata
    add columns to the df
        yhat,
        baseline,
        esidual,
        baseline_residual,
        residual_sqr,
        baseline_residual_sqr
    takes in and solves SSE, ESS, TSS, MSE, and RMSE
    and returns them as well'''
    # get the data
    tips = data("tips")
    # fit the model
    model = ols('total_bill ~ tip', data=tips).fit()
    # create new features
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
    tips['residual_sqr'] = tips.residual ** 2
    tips['baseline_residual_sqr'] =  tips.baseline_residual ** 2
        # SSE
    SSE = tips['residual_sqr'].sum()
    SSE_baseline =  tips['baseline_residual_sqr'].sum()
        # TSS
    TSS = SSE_baseline =  tips['baseline_residual_sqr'].sum()
        # ESS
    ESS = TSS - SSE
        # MSE
    MSE = SSE / len(tips)
        # RMSE
    RMSE = sqrt(MSE)
    
    print("My Sum of squared error is:")
    print(" ")
    print(SSE) 
    print("----------------------------------------------")
    print("My Total Sum of Square is:")
    print(" ")
    print(TSS) 
    print("----------------------------------------------")
    print("My Explained sum of squares is:")
    print(" ")
    print(ESS) 
    print("----------------------------------------------")
    print("My Mean of Square Error Values are:")
    print(" ")
    print(MSE)
    print("----------------------------------------------")
    print("My Root Mean of Square Error Values are:")
    print(" ")
    print(RMSE)
    
def baseline_mean_errors():
    '''
    gather tips data set from pydata
    add columns to the df
        yhat,
        baseline,
        esidual,
        baseline_residual,
        residual_sqr,
        baseline_residual_sqr
    Take sin SSE_baseline, MSE_baseline, and RMSE_baseline and returns them
    '''
    # get the data
    tips = data("tips")
    # fit the model
    model = ols('total_bill ~ tip', data=tips).fit()
    # create new features
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
    tips['residual_sqr'] = tips.residual ** 2
    tips['baseline_residual_sqr'] =  tips.baseline_residual ** 2
        # SSE_baseline
    SSE_baseline =  tips['baseline_residual_sqr'].sum()   
        # MSE_baseline
    MSE_baseline = SSE_baseline / len(tips)    
        # RMSE_baseline
    RMSE_baseline = sqrt(MSE_baseline)
    print("Baseline of Sum of Square Error Values are:")
    print(" ")
    print(SSE_baseline) 
    print("----------------------------------------------")
    print("Baseline of Mean of Square Error Values is:")
    print(" ")
    print(MSE_baseline)
    print("----------------------------------------------")
    print("Baseline of Root Mean of Square Error Values is:")
    print(" ")
    print(RMSE_baseline) 
    
def better_than_baseline():
    '''
    gather tips data set from pydata
    add columns to the df
        yhat,
        baseline,
        esidual,
        baseline_residual,
        residual_sqr,
        baseline_residual_sqr
    Make evs
    and return true if evs is greater then baseline false if not
    '''
    # get the data
    tips = data("tips")
    # fit the model
    model = ols('total_bill ~ tip', data=tips).fit()
    # create new features
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
    tips['residual_sqr'] = tips.residual ** 2
    tips['baseline_residual_sqr'] =  tips.baseline_residual ** 2
    # set the evs
    evs = explained_variance_score(tips.total_bill, tips.yhat)
    # set baseline
    baseline = tips.total_bill.mean()
    # return
    if evs > baseline:
        return True
    else:
        return False