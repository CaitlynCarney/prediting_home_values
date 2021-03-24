import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import seaborn as sns

# modeling methods
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression 

import warnings
warnings.filterwarnings("ignore")
import acquire
import prepare
import explore
import evaluate

def xtrain_xval_xtest():
    '''create X_train, X_validate, X_test, y_train, y_validate, y_test'''
    # get the data
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # split the data
    train, validate, test = prepare.split_focused_zillow(df)
    X_train = train.drop(columns = ['appraised_value'])
    y_train = train[['appraised_value']]

    X_validate = validate.drop(columns=['appraised_value'])
    y_validate = validate[['appraised_value']]

    X_test = test.drop(columns=['appraised_value'])
    y_test = test[['appraised_value']]
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_baseline():
    ''' takes in data and sets the baseline for the model'''
    # get data
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    # make into data frames
    y_train = pd.DataFrame(y_train)
    # turn it into a single pandas dataframe
    y_validate = pd.DataFrame(y_validate)
    # 1. Predict appraised_value_pred_mean
        # 2 different aselines of mean and medium
    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean
        #compute appraised_value_pred_median
            # same process as mean (above)
    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median
        # RMSE of appraised_value_pred_mean
    rmse_train_mean = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_mean)**(1/2)
        # medium
    rmse_train_median = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_median)**(1/2)

    rmse_validate_median = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_median)**(1/2)
        # do the same thing for the validate set as done above for the train set
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train_mean, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate_mean, 2))
    print(' ')

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train_median, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate_mean, 2))

def all_models_info():
    '''takes in data
    sets baseline
    sets SSE, MSE, and RMSE
    returns infor for all 4'''
    # get data
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
    #LARS Model
    lars = LassoLars(alpha=1.0)
    lars.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lars'] = lars.predict(X_train)
    rmse_train_lars = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lars)**1/2
    y_validate['appraised_value_pred_lars'] = lars.predict(X_validate)
    rmse_validate_lars = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lars)**1/2
    #GLM
    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_glm'] = glm.predict(X_train)
    rmse_train_glm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_glm)**1/2
    y_validate['appraised_value_pred_glm'] = glm.predict(X_validate)
    rmse_validate_glm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_glm)**1/2
    # PF
    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train)
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)
    # LM2
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train.appraised_value)
    y_train['appraised_value_pred_lm2'] = lm2.predict(X_train_degree2)
    rmse_train_lm2 = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm2)**1/2
    y_validate['appraised_value_pred_lm2'] = lm2.predict(X_validate_degree2)
    rmse_validate_lm2 = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm2)**1/2
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_lm, 
          "\nValidation/Out-of-Sample: ", rmse_validate_lm)
    print("--------------------------------------------------------------")
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train_lars, 
          "\nValidation/Out-of-Sample: ", rmse_validate_lars)
    print("--------------------------------------------------------------")
    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train_glm, 
          "\nValidation/Out-of-Sample: ", rmse_validate_glm)
    print("--------------------------------------------------------------")
    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train_lm2, 
          "\nValidation/Out-of-Sample: ", rmse_validate_lm2)
    
def plot_actual_and_pred():
    '''takes in data from all_models_info
    plots the actual appraised_value and the predicted appraised_value'''
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    # Baseline
    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean
        #compute appraised_value_pred_median
            # same process as mean (above)
    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median
    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
    # Make the plot
    plt.figure(figsize=(20,10))
    sns.set(style="darkgrid")

    plt.scatter(y_validate.appraised_value, y_validate.appraised_value_pred_lm, 
                alpha=.5, color="mediumblue", s=100, label="Model: LinearRegression")
    m, b = np.polyfit(y_validate.appraised_value, y_validate.appraised_value_pred_lm, 1)
    plt.plot(y_validate.appraised_value, m*y_validate.appraised_value+b, color='limegreen', label='Line of Regrssion', linewidth=5)
    plt.plot(y_validate.appraised_value, y_validate.appraised_value_pred_mean, alpha=.5, color="yellow", label='Baseline', linewidth=5)
    plt.plot(y_validate.appraised_value, y_validate.appraised_value, alpha=.5, color="cyan", label='The Ideal Line: Predicted = Actual', linewidth=5)
    plt.title('Plotting Actual vs. Predicted Values')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
def plot_ols_errors():
    '''takes in data from all_models_info
    plots the errors of the model'''
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    # Baseline
    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean
        #compute appraised_value_pred_median
            # same process as mean (above)
    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median
    appraised_value_pred_median = y_train['appraised_value'].median()
    rmse_train_mean = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_mean)**(1/2)
    rmse_train_median = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_median)**(1/2)
    rmse_validate_median = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_median)**(1/2)
    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
    # Make the plot
    plt.figure(figsize=(20,10))
    sns.set(style="darkgrid")

    plt.scatter(y_validate.appraised_value, y_validate.appraised_value_pred_lm-y_validate.appraised_value, 
                alpha=.5, color="mediumblue", s=100, label="Model: LinearRegression")
    plt.axhline(label="No Error", color='black', linewidth=7)
    plt.title('Plotting the Errors in Predictions')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
def hist_ols_appraised_value():
    '''takes in data from all_models_info
    plots histograms of actual and predicted appraised_value'''
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    # Baseline
    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean
        #compute appraised_value_pred_median
            # same process as mean (above)
    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median
    appraised_value_pred_median = y_train['appraised_value'].median()
    rmse_train_mean = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_mean)**(1/2)
    rmse_train_median = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_median)**(1/2)
    rmse_validate_median = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_median)**(1/2)
    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
    # Create histograms
    plt.subplots(3, 5, figsize=(8,16), sharey=True)
    sns.set(style="darkgrid")
    plt.title("Comparing the Distribution of appraised_values to Distributions of Predicted appraised_values Linear Regression Models")
    plt.xlabel("appraised_value", size = 15)
    plt.ylabel("appraised_value Count", size = 15)

    plt.subplot(3,1,1)
    plt.hist(y_validate.appraised_value, color='cyan', alpha=.5,  ec='black')
    plt.title('Actual appraised_values', size=15)

    plt.subplot(3,1,2)
    plt.hist(y_validate.appraised_value_pred_lm, color='lawngreen', alpha=.5,  ec='black')
    plt.title('Model: LinearRegression', size=15)

    plt.subplot(3,1,3)
    plt.hist(y_validate.appraised_value, color='lawngreen', alpha=.5, label="Actual Final appraised_values", ec='black')
    plt.hist(y_validate.appraised_value_pred_lm, color='cyan', alpha=.5, label="Model: LinearRegression", ec='black')
    plt.title("All Graphs Stacked", size=15)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    
    
def choose_best_model():
    '''takes in data from all_models_info
    and choosed which model should move forwards as the best model
    this model will be ran using the test data'''
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    # pull from add to trian
    train = evaluate.add_to_train()
    X_train, y_train, X_validate, y_validate, X_test, y_test = evaluate.xtrain_xval_xtest()
    # make into data frames
    y_train = pd.DataFrame(y_train)
    # turn it into a single pandas dataframe
    y_validate = pd.DataFrame(y_validate)
    # Predict appraised_value_pred_mean
    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean
    #compute appraised_value_pred_median
    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median
    # RMSE of appraised_value_pred_mean
    rmse_train_mean = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_mean)**(1/2)
    # OLS mode
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.appraised_value)
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
        # make sure you are using x_validate an not x_train
    # Make the choice
    print("Model Selected: RMSE for OLS using Linear Regression")
    print("--------------------------------------------------------------")  
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train_mean, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate_mean, 2))
    print("--------------------------------------------------------------")
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_lm, 
          "\nValidation/Out-of-Sample: ", rmse_validate_lm)
    
def test_final_model():
    df = acquire.acquire_zillow()
    df = prepare.clean_zillow(df)
    df = prepare.focused_zillow(df)
    train, validate, test = prepare.split_focused_zillow(df)
    
    X_train = train.drop(columns = ['appraised_value'])
    y_train = train[['appraised_value']]

    X_validate = validate.drop(columns=['appraised_value'])
    y_validate = validate[['appraised_value']]

    X_test = test.drop(columns=['appraised_value'])
    y_test = test[['appraised_value']]
    y_train = pd.DataFrame(y_train)

    y_validate = pd.DataFrame(y_validate)

    appraised_value_pred_mean = y_train['appraised_value'].mean()
    y_train['appraised_value_pred_mean'] = appraised_value_pred_mean
    y_validate['appraised_value_pred_mean'] = appraised_value_pred_mean

    appraised_value_pred_median = y_train['appraised_value'].median()
    y_train['appraised_value_pred_median'] = appraised_value_pred_median
    y_validate['appraised_value_pred_median'] = appraised_value_pred_median

    rmse_train_mean = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_mean)**(1/2)

    rmse_validate_mean = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_mean)**(1/2)
    
    # sert up the model
    lm = LinearRegression(normalize=True)
    # fit the model
    lm.fit(X_train, y_train.appraised_value)
    # predict train
    y_train['appraised_value_pred_lm'] = lm.predict(X_train)
    # evaluate: rmse
    rmse_train_lm = mean_squared_error(y_train.appraised_value, y_train.appraised_value_pred_lm)**(1/2)
    # predict validate
    y_validate['appraised_value_pred_lm'] = lm.predict(X_validate)
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.appraised_value, y_validate.appraised_value_pred_lm)**(1/2)
        # make sure you are using x_validate an not x_train
    # test the model
    lm = LinearRegression(normalize=True)
    lm.fit(X_test, y_test.appraised_value)
    y_test['appraised_value_pred_lm'] = lm.predict(X_test)
    rmse_test_lm = mean_squared_error(y_test.appraised_value, y_test.appraised_value_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression Test Data: ", rmse_test_lm)
