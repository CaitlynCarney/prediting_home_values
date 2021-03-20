import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import env

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer


def clean_zillow(df):
    '''
    clean_zillow will take in df from acquire_zillow, and return cleaned pandas dataframe
    will drop all columns with: 
        less than 35,000 non null values
        needs 90.72% non null values
    will remove features:
        'calculatedbathnbr'
        'finishedsquarefeet12'
        'propertycountylandusecode'
        'logerror'
        'transactiondate'
        'yearbuilt'
        'taxvaluedollarcnt'
        'landtaxvaluedollarcnt'
        'rawcensustractandblock'
    rename 'bathroomcnt' to 'bathrooms'
    rename 'bedroomcnt' to 'bedrooms'
    rename 'calculatedfinishedsquarefeet'to 'square_feet'
    rename 'fullbathcnt' to 'full_baths'
    rename 'regionidzip' to 'zip_code'
    rename 'regionidcity' to 'city'
    rename 'regionidcounty' to 'county'
    '''
    #drop all the features with less than 35,000 non null values
    df = df.dropna(axis=1,thresh=35000)
    # drop unneeded columns
    df = df.drop(['calculatedbathnbr', 'finishedsquarefeet12', 
             'propertycountylandusecode', 'logerror', 'transactiondate',  
             'yearbuilt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 
              'rawcensustractandblock', 'censustractandblock', 
              'structuretaxvaluedollarcnt', 'parcelid', 'id'], axis=1)
    # rename the columns needed
    df = df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 
                       'calculatedfinishedsquarefeet':'square_feet', 
                       'fullbathcnt':'full_baths', 'regionidzip':'zip_code', 
                       'regionidcity':'city', 'regionidcounty':'county'})
    # drop nulls
    df = df.dropna()
    # this ends up dropping from 38582 to 37712 
        # we lost 870 rows by dropping
    return df


def focused_zillow(df):
    '''
    takes in clean_zillow
    sets sepecific features to focus on

    returns a focused data frame in a pandas dataframe
    '''
    features = [
    'square_feet',
    'bedrooms',
    'bathrooms',
    'zip_code',
    'propertylandusetypeid',
    'taxamount']
    df = df[features]
    return df

def split_zillow(df):
    '''
    splt_iris will take one argument df, a pandas dataframe, anticipated to be the telco dataset
    sets sepecific features to focus on
    sets index
    replace all blank cells with null values
    drop all nulls in the df
    change 'total_charges' dtype from object to float
    
    perform a train, validate, test split
    
    return: three pandas dataframes: train, validate, test
    '''
    
    df = clean_telco(df)
    train, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train, test_size=.3, 
                                       random_state=1234)
    return train, validate, test

def min_max_scaler(train, validate, test):
    '''
    take in split_zillow df
    scales the df using 'MinMaxScaler'
        makes the scaler object
        fits onto train set
        uses
    returns scaled df
    '''
    df = split_telco(train, validate, test)
    # Step 1 Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Fit the thing
    scaler.fit(train)
    # Create Train Valideate and test sample sets
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    return train_scaled, validate_scaled, test_scaled

def quantile_transformer(train, validate, test):
    '''
    take in split_telco df
    scales the df using 'MinMaxScaler'
        makes the scaler object
        fits onto train set
        uses
    returns scaled df
    '''
    df = split_telco(train, validate, test)
    # Step 1 Make the thing
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    # Fit the thing
    scaler.fit(train)
    # Create Train Valideate and test sample sets
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    return train_scaled, validate_scaled, test_scaled
