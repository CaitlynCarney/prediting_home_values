import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import env

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_telco(df):
    '''
    This function reads the telco_churn data from the Codeup db into a df,
    and returns the df.
    '''
    query = """
    select * 
    from customers;
    """
    df = pd.read_sql(query, get_connection('telco_churn'))
    return df


def clean_telco(df):
    '''
    takes in dataframe
    sets sepecific features to focus on
    sets index
    replace all blank cells with null values
    drop all nulls in the df
    change 'total_charges' dtype from object to float

    returns clean data frame in a pandas dataframe
    '''
    df = get_telco(df)
    features = [
    'customer_id',
    'tenure',
    'monthly_charges',
    'total_charges'
    ]
    df = df[features]
    df = df.set_index("customer_id")
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.dropna()
    df['total_charges'] = df.total_charges.astype('float')
    return df

def split_telco(df):
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
    take in split_telco df
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
