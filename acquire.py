from env import host, user, password
import pandas as pd
import numpy as np

def acquire_all_telco():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select *
    from customers
    join contract_types using(contract_type_id)
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
    df = pd.read_sql(sql_query, connection)
    return df

def acquire_telco():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select customer_id, monthly_charges, tenure, total_charges
    from customers
    join contract_types using(contract_type_id)
    where contract_type = "Two Year"
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
    df = pd.read_sql(sql_query, connection)
    return df

def acquire_zillow():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select *
    from properties_2017
    join predictions_2017 using(parcelid)
    where propertylandusetypeid = 260
    or 261 or 263 or 273 or 274 or 276 or 279
    and unitcnt = 1
    and transactiondate 
    like '2017-05-%%' 
    or '2017-06-%%';
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    df = pd.read_sql(sql_query, connection)
    return df
