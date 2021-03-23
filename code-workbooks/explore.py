import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from math import sqrt
from statsmodels.formula.api import ols

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 

from env import host, user, password
import acquire
import prepare

def plot_zillow_heatmap():
    '''Plots heatmap of cleaned zillow dataset'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # plot the heatmap
    plt.figure(figsize=(16, 6))
    corr_map = sns.heatmap(df.corr(), cmap="viridis", vmin=-1, vmax=1, annot=True)
    corr_map.set_title('Zillow Correlation Heatmap of Zilllow Data', fontdict={'fontsize':18}, pad=12)

def plot_train_heatmap():
    '''Plots heatmap of split cleaned zillow dataset'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Plot the heatmap
    plt.figure(figsize=(16, 6))
    sns.heatmap(train.corr(), cmap="viridis", vmin=-1, vmax=1, annot=True, 
                           center=0, linewidths=4, linecolor='silver')
    plt.title('Zillow Correlation Heatmap of Trained Data without Scaling', fontsize=18, pad=12)

def plot_scatter_plots():
    '''Plots scatter plots to show the relationships between features and appraised_value'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Plot
    plt.subplots(3, 1, figsize=(12,40), sharey=True)
    sns.set(style="darkgrid")

    plt.subplot(3,1,1)
    plt.title("Relationship between Appraised Values, and Bathrooms", size=20, color='black')
    sns.scatterplot(data=train, x='appraised_value', y='bathrooms', hue='bathrooms', palette='viridis')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.subplot(3,1,2)
    plt.title("Relationship between Appraised Values, Bedrooms", size=20, color='black')
    sns.scatterplot(data=train, x='appraised_value', y='bedrooms', hue='bedrooms', palette='viridis')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.subplot(3,1,3)
    plt.title("Relationship between Appraised Values, Square Footage of Homes", size=20, color='black')
    sns.scatterplot(data=train, x='appraised_value', y='square_feet', hue='square_feet', palette='viridis')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

def train_pairplot():
    '''Plots histograms of each feature'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Plot
    sns.pairplot(train, hue = 'appraised_value', palette='viridis')

def bathroom_corr():
    '''Runs correlation test between bathrooms and appraised value
    plot distribution plot
    plot box plot'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Set nul and alternative hypothesis, confidence level, and alpha
    null_hypothesis = "There is no correlation between number of bathrooms and appraised value."
    alt_hypothesis = "There is a correlation between number of bathrooms and appraised value."
    confidence_level = .95
    a = 1 - confidence_level
    # set x and y
    x = train.bathrooms
    y= train.appraised_value
    # run it
    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between Bathrooms and the Appraised value is: ', corr)
    print(f' The P value between Bathrooms and Appraised Value is: ', p)
    print(' ')
    if p < a:
        print(f"Reject null hypothesis: '{null_hypothesis}'")
        print(' ')
        print(f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
        print(' ')
        if 0 < corr < .6:
            print("This is a weak positive correlation.")
        elif .6 < corr < 1:
            print("That is a strong positive correlation.")
        elif -.6 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.6:
            print("That is a strong negative correlation.")
    
    else : 
        print("Fail to reject the null hypothesis.")
    # distplot
    sns.distplot(train.bathrooms, kde=True, color='teal')
    # boxplot
    sns.boxplot(y='appraised_value', x ='bathrooms', data = train, palette='viridis')
    
def bedroom_corr():
    '''Runs correlation test between bedrooms and appraised value
    plot a distribution plot
    plot a boxplot'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Set nul and alternative hypothesis, confidence level, and alpha
    null_hypothesis = "There is no correlation between number of bedrooms and appraised value."
    alt_hypothesis = "There is a correlation between number of bedrooms and appraised value."
    confidence_level = .95
    a = 1 - confidence_level
    x = train.bedrooms
    y= train.appraised_value

    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between Bathrooms and the Appraised value is: ', corr)
    print(f' The P value between Bathrooms and Appraised Value is: ', p)
    print(' ')
    if p < a:
        print(f"Reject null hypothesis: '{null_hypothesis}'")
        print(' ')
        print(f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
        print(' ')
        if 0 < corr < .6:
            print("This is a weak positive correlation.")
        elif .6 < corr < 1:
            print("That is a strong positive correlation.")
        elif -.6 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.6:
            print("That is a strong negative correlation.")

    else : 
        print("Fail to reject the null hypothesis.")
    # distplot
    sns.distplot(train.bedrooms, kde=True, color='teal')
    # boxplot
    sns.boxplot(y='appraised_value', x ='bedrooms', data = train, palette='viridis')
    

def square_feet_corr():
    '''Runs correlation test between bedrooms and appraised value
    plot a distribution plot
    plot a jointplot'''
    # Take in the dataframe
    df = acquire.acquire_zillow()
    # Prepare the data
    df = prepare.clean_zillow(df)
    # Split the data set
    train, validate, test = prepare.split_focused_zillow(df)
    # Set nul and alternative hypothesis, confidence level, and alpha
    null_hypothesis = "There is no correlation between a homes square footage and appraised value."
    alt_hypothesis = "There is a correlation between square feet and appraised value."
    confidence_level = .95
    a = 1 - confidence_level
    x = train.square_feet
    y= train.appraised_value

    corr, p = stats.pearsonr(x, y)
    print(f' The correlation between Bathrooms and the Appraised value is: ', corr)
    print(f' The P value between Bathrooms and Appraised Value is: ', p)
    print(' ')
    if p < a:
        print(f"Reject null hypothesis: '{null_hypothesis}'")
        print(' ')
        print(f"We now move forward with our alternative hypothesis: '{alt_hypothesis}'")
        print(' ')
        if 0 < corr < .6:
            print("This is a weak positive correlation.")
        elif .6 < corr < 1:
            print("That is a strong positive correlation.")
        elif -.6 < corr < 0:
            print("This is a weak negative correlation.")
        elif -1 < corr < -.6:
            print("That is a strong negative correlation.")

    else : 
        print("Fail to reject the null hypothesis.")
    #distplot
    sns.distplot(train.square_feet, kde=True, color='teal')
    #jointplot
    sns.jointplot(data = train, x = 'square_feet', y = 'appraised_value', color='teal')