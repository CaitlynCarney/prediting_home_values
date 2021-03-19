from env import host, user, password
import pandas as pd
import numpy as np


def acquire_zillow():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select *
    from  properties_2017
    join predictions_2017 using(parcelid)
    where transactiondate between "2017-05-01" and "2017-08-31"
        and propertylandusetypeid between 260 and 266
            or propertylandusetypeid between 273 and 279
            and not propertylandusetypeid = 274
        and unitcnt = 1;
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    df = pd.read_sql(sql_query, connection)
    return df
