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
    will create new feature:
        `house_age`
            takes the `yearbuilt` feature and subtracts it from the current year '2021'
        `tax_rate`
            takes the `taxvaluedollarcnt` and divides it by `taxamount`
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
    handle outliers using IQR rule for the features:
        `appraised_value`
        `square_feet`
        `bedrooms`
    '''
    #create new feature house_age
    df['house_age'] = 2021 - df.yearbuilt
    #create new feature tax_rate which is the monthyl taxes
    df['tax_rate'] = df.taxvaluedollarcnt / df.taxamount
    #drop all the features with less than 35,000 non null values
    df = df.dropna(axis=1,thresh=35000)
    # drop unneeded columns
    df = df.drop(['calculatedbathnbr', 'finishedsquarefeet12', 
             'propertycountylandusecode', 'logerror', 'transactiondate',  
             'yearbuilt', 'landtaxvaluedollarcnt', 
              'rawcensustractandblock', 'censustractandblock', 
              'structuretaxvaluedollarcnt', 'parcelid', 'id'], axis=1)
    # rename the columns needed
    df = df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 
                       'calculatedfinishedsquarefeet':'square_feet', 
                       'fullbathcnt':'full_baths', 'regionidzip':'zip_code', 
                       'regionidcity':'city', 'regionidcounty':'county',
                       'taxvaluedollarcnt':'appraised_value', 
                        'propertylandusetypeid':'house_type'})
    # drop nulls
    df = df.dropna()
    # this ends up dropping from 38582 to 37712 
        # we lost 870 rows by dropping
    df = df[df.appraised_value <= 1_123_603.75]
    # df have gone from 37,711 rows to 35,187 rows
        # the df has lost 2,524 rows
    df = df[df.square_feet <= 3_131.5]
    # df have gone from 35,187 rows to 33,849 rows
        # the df has lost 1,338 rows
    df = df[df.bedrooms <= 4.5]
    df = df[df.bedrooms >= 0.5]
    # df have gone from 33,849 rows to 32,538 rows
        # the df has lost 1,311 rows
    return df

def split_clean_zillow(df):
    '''
    splt_zillow will take one argument df, a pandas dataframe, anticipated to be the telco dataset
    sets sepecific features to focus on
    sets index
    replace all blank cells with null values
    drop all nulls in the df
    change 'total_charges' dtype from object to float
    
    perform a train, validate, test split
    
    return: three pandas dataframes: train, validate, test
    '''
    
    df = clean_zillow(df)
    train, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train, test_size=.3, 
                                       random_state=1234)
    return train1, validate1, test1


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
    'appraised_value']
    df = df[features]
    return df

def split_focused_zillow(df):
    '''
    splt_zillow will take one argument df, a pandas dataframe, anticipated to be the telco dataset
    sets sepecific features to focus on
    sets index
    replace all blank cells with null values
    drop all nulls in the df
    change 'total_charges' dtype from object to float
    
    perform a train, validate, test split
    
    return: three pandas dataframes: train, validate, test
    '''
    
    df = focused_zillow(df)
    train_validate, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=1234)
    return train, validate, test

def scale_focused_zillow(train, validate, test):
    '''scale_focused_zillow will take train, validate, and test from split_focused_zillow
    create a scaler
    fit scaler
    create train_scaled, validate_scaled, and test_scaled
    and turn each of them into data frames
    
    return: three pandas dataframes: train_scaled, validate_scaled, test_scaled
    '''
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    # fit the thing
    scaler.fit(train)

    # tun them
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # hey pandas make them into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    # return them
    
    return train_scaled, validate_scaled, test_scaled


def min_max_scaler(train, validate, test):
    '''
    take in split_zillow df
    scales the df using 'MinMaxScaler'
        makes the scaler object
        fits onto train set
        uses
    returns scaled df
    '''
    train, validate, test = split_focused_zillow(df)
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    # fit the thing
    scaler.fit(train)
    
    X_train = train.drop(columns = ['appraised_value'])
    y_train = train.appraised_value
    X_validate = validate.drop(columns=['appraised_value'])
    y_validate = validate.appraised_value
    X_test = test.drop(columns=['appraised_value'])
    y_test = test.appraised_value

    # tun them
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.transform(y_train)
    y_validate_scaled = scaler.transform(y_validate)
    y_test_scaled = scaler.transform(y_test)    
    
    # hey pandas make them into dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train.columns)
    X_validate_scaled = scaler.transform(X_validate_scaled, columns=train.columns)
    X_test_scaled = scaler.transform(X_test_scaled_scaled, columns=train.columns)
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=train.columns)
    y_validate_scaled = scaler.transform(y_validate_scaled, columns=train.columns)
    y_test_scaled = scaler.transform(y_test_scaled, columns=train.columns)
    # return them
    
    return train_scaled, validate_scaled, test_scaled, X_train_scaled, X_validate_scaled, X_test_scaled, y_train_scaled, y_validate_scaled, y_test_scaled
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
