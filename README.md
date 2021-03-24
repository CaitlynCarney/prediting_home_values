# <a name="top"></a>Zillow® Project - README.md
![](http://zillow.mediaroom.com/image/Zillow_Wordmark_Blue_RGB.jpg)

[[Data Dictionary](#dictionary)]
[[Project Description](#project_description)]
[[Project Planning](#project_planning)]
[[Project Acquire](#project_acquire)]
[[Project Prepare](#project_prepare)]
[[Project Explore](#project_explore)]
[[Key Findings](#findings)]
[[Data Acquire, Prep, and Exploration](#wrangle)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]

## Trello Board Link
https://trello.com/b/G0AAKAdh/regression-project

## Linkedin Link
https://www.linkedin.com/in/caitlyn-carney-a29b241aa/


## Data Dictonary
<a name="dictionary"></a>
[[Back to top](#top)]

| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| appraised_value* | The total tax assessed value of the parcel | float |
| bathrooms | Number of bathrooms in home including fractional bathrooms | float |
| bedrooms | Number of bedrooms in home | float |
| city |   City in which the property is located (if any) | int |
| county |   County in which the property is located) | int |
| fips |   Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details | int |
| house_age | year_built minus 2021 | int |
| house_type |  Type of land use the property is zoned for | int |
| latitude | Latitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |
| longitude | Longitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |
| parcelid | Unique identifier for parcels (lots) | Index/int | 
| square_feet | Calculated total finished living area of the home | float |
| taxamount	|  The total property tax assessed for that assessment year | int |
| untcnt |   Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...) | int |
| zip_code |  Zip code in which the property is located | int |

## Project Description and Goals
<a name="project_description"></a>

- Project description:
    - I need to be able to predict the values of single unit properties that the tax district assesses using the property data from those with a transaction between May-August, 2017.
        - I also need to know what states and counties these are located in, the distribution of tax rates for each county
        - I will pull data from the codeup sequel server.

- My goal is to:
    - Goal is to predict a homes value
    - but what are you defining as the homes value
    - what the county assesses the county at?


# Project Planning
## <a name="project_planning"></a>
[[Back to top](#top)]

 **Plan** -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

- Tasking out how I plan to work through the pipeline.

### Target variable
- appraised_value

### Starting focus features
- bedrooms
- bathrooms
- square_feet

### For Second Run Through
- zip_code
- house_age
- tax_rate
- bedrooms
- bathrooms
- square_feet

### Projet Outline:
- Acquisiton of data through Codeup SQL Server
- Prepare and clean data using python
- Explore data
    - Remove features
        - too many nulls?
        - not helpful to the quest?
    - Create features as needed
    - Handle null values
        - are the fixable or should they just be deleted
    - Handle outliers
    - Make graphs that show 
- Run statistical analysis
- Model data 
- Test Data
- Conclude results

### Hypotheses
- City, county, and zip code drive home pricing
- Number of Bedrooms drives pricing
- Number of bathrooms drive pricing
- Homes with a central unit ac (id =1) have a higher price than others
- The higher the square footage the higher the price

# Project Acquiring
<a name="project_acquire"></a>
[[Back to top](#top)]

 Plan -> **Acquire** -> Prepare -> Explore -> Model & Evaluate -> Deliver

Functions used can be found in acquire.py in git hub repo

1. acquire the zillow data from the codeup sequel server and convert it into a pandas df
    `def acquire_zillow():
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
    return df`

2. check out the .info
    - Takeways
        - There are many columns who has more that 10- 100% of their rows filled with null values
            - Need to either fix or drop these columns
        - There are 61 columns
            - I am sure we can dwindle that down to a more managable size
3. check out the .describe
    - Everything looks good


# Project Preperation
<a name="project_prepare"></a>
[[Back to top](#top)]

 Plan -> Acquire -> **Prepare** -> Explore -> Model & Evaluate -> Deliver

Functions used can be found in prepare.py in git hub repo

1. clean the data:
    ```def clean_zillow(df):
        #drop all the features with less than 35,000 non null values
        df = df.dropna(axis=1,thresh=35000)
        # drop unneeded columns
        df = df.drop(['calculatedbathnbr', 'finishedsquarefeet12', 
                 'propertycountylandusecode', 'logerror', 'transactiondate',  
                 'yearbuilt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                  'rawcensustractandblock', 'censustractandblock', 
                  'structuretaxvaluedollarcnt'], axis=1)
        # rename the columns needed
        df = df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 
                           'calculatedfinishedsquarefeet':'square_feet', 
                           'fullbathcnt':'full_baths', 'regionidzip':'zip_code', 
                           'regionidcity':'city', 'regionidcounty':'county'})
        return df```
2. focus on 5 main features
    ```def focused_zillow(df):
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
    'propertylandusetypeid']
    df = df[features]
    return df```
3. check out the new .info
    - Takeaways:
        - We have 37,711 entries/rows
            - non of which ar null or blank
        - All feature/columns are floats meaning they are numeric but have at least one decimal.
        - our features are:
            - `square_feet`
            - `bedrooms`
            - `bathrooms`
            - `city`
            - `house_type`
            - `appraised_value`
4. check out the new .describe
    - Takeaways
        - Averages
            - `square_feet` = ~1,754.38
            - `bedrooms` = ~ 3
            - `bathrooms` = ~ 2.28
            - `appraised_value` = ~ 49,3804.6
5. check out the value_counts
6. run a df.isna().sum()
    - 0
7. run a df.isnull().sum()
    - 0 
8. split the data

    - spliting the clean_zillow dataset
        ```def split_clean_zillow(df):
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
        return train1, validate1, test1```
    - spliting the focused_zillow dataset
        ```'''splt_zillow will take one argument df, a pandas dataframe, anticipated to be the telco dataset
        sets sepecific features to focus on
        sets index
        replace all blank cells with null values
        drop all nulls in the df
        change 'total_charges' dtype from object to float

        perform a train, validate, test split

        return: three pandas dataframes: train, validate, test
        '''

        df = focused_zillow(df)
        train, test = train_test_split(df, test_size=.2, random_state=1234)
        train, validate = train_test_split(train, test_size=.3, 
                                           random_state=1234)
        return train, validate, test```
9. Scale the split data
        ```scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(train)
        train_scaled = scaler.transform(train)
        validate_scaled = scaler.transform(validate)
        test_scaled = scaler.transform(test)
        train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
        validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
        test_scaled = pd.DataFrame(test_scaled, columns=train.columns)```

# Project Exploration
<a name="project_explore"></a>
[[Back to top](#top)]

 Plan -> Acquire -> Prepare -> **Explore** -> Model & Evaluate -> Deliver

1. pull my acquire
    `df = acquire.acquire_zillow()`
2. pull my prepare and prepare_telco
    `df = prepare.clean_zillow(df)`
    
    `df = prepare.focused_zillow(df)``
3. split my data
    `train, validate, test = prepare.split_focused_zillow(df)`
4. create correlation heat map
    `corr_map = sns.heatmap(df.corr(), cmap="viridis", vmin=-1, vmax=1, annot=True)`
5. Create histograms for features being used
    `train.hist(bins=50, figsize=(30,20), color='teal', ec='black');`
6. Run a stat test for each feature
7. Answering my questions:  


# Stat Tests
## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

Correlation Test
 - Used to check if two samples are related. They are often used for feature selection and multivariate analysis in data preprocessing and exploration.

