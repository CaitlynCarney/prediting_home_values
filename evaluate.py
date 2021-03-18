import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from statsmodels.formula.api import ols

from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pydataset import data
import acquire


def plot_residuals():
    ''' Gather tips data set from pydata
    add columns yhat, baseline, residualm and baseline residual to the df
    plot residual scatterplot
    plot baseline residual scatterplots'''
    tips = data("tips")
    model = ols('total_bill ~ tip', data=tips).fit()
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
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
    tips = data("tips")
    model = ols('total_bill ~ tip', data=tips).fit()
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
    tips = data("tips")
    model = ols('total_bill ~ tip', data=tips).fit()
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
    tips = data("tips")
    model = ols('total_bill ~ tip', data=tips).fit()
    tips['yhat'] = model.predict(tips.tip)
    yhat = tips.yhat
    tips['baseline'] = tips.total_bill.mean()
    tips['residual'] = tips.total_bill - tips.yhat
    tips['baseline_residual'] = tips.total_bill - tips.baseline
    tips['residual_sqr'] = tips.residual ** 2
    tips['baseline_residual_sqr'] =  tips.baseline_residual ** 2
    evs = explained_variance_score(tips.total_bill, tips.yhat)
    baseline = tips.total_bill.mean()
    if evs > baseline:
        return True
    else:
        return False