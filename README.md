# Project Planning
 **Plan** -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

- Project description:
    - By using what I have learned in the Regression Module:
        - data acquisition
        - data preparation
        - scaling
        - exploratory analysis
        - evaluation
        - feature engineering
        - and modeling
            - I will be showing my code, findings, models, key takeaways, and recommendations regarding home pricing.
    - I am a Zillow data scientist
        - I need to be able to predict the values of single unit properties that the tax district assesses using the property data from those with a transaction between May-August, 2017.
        - I also need to know what states and counties these are located in, the distribution of tax rates for each county
        - I will pull data from the codeup sequel server.

- My goal is to:
    - Goal is to predict a homes value
    - but what are you defining as the homes value
    - what the county assesses the county at?

- My deliverable:   
    1.  When I have a solid presentation which:
        - Summarize my findings about the drivers
        - Visualizations supporting my main points.
        - Should have:
            - Title slide
            - Agenda slide
            - Executive summary slide. 
                - Your Big Idea, 
                - Goal, 
                - key finding(s), 
                - recommendation).
            - Overview about your data sample to give context...
                - who are they? 
                - are they a subsample of the original? 
                - How many people are churning over all? 
                    - Establish the problem. 
                    - Yes, we have an issue! 
            - Visualization of the issue you found 
            - Recommendation you are making
            - Visualization of changes expected if action is taken.
            - Conclusion and next steps (“with more time, I would like to…”, e.g.)
            - Appendix slides are included with info such as…
                - data definitions.
                -  sample explanations (i.e. the subset, the types of 
                                customers (month-to-month, internet only customers, 
                                e.g.)).
                    -  an aggregate table of summary data in the background of                                                                                      
                    - your chart(s) (this can be created easily in tableau...right  
                    - click on chart and say "create crosstab").
    2. A github repository containing your work.
        - Containing
            - one-two clearly labeled final Jupyter Notebook that walks through the pipeline
            - In exploration, I will perform my analysis including: 
                - the use of at least two statistical tests 
                - visualizations documenting hypotheses 
                - takeaways. 
            - In modeling, I will
                - establish a baseline that I will attempt to beat with various  
                   algorithms and/or hyperparameters. 
                - Evaluating my model by computing the metrics and comparing.
        - Make sure your notebook answers all the questions posed by the  
           Zillow data science team.
        - The repository should also contain the 
            - .py files necessary to reproduce your work, 
                - acquire.py
                - prepare.py
        - an README.md file documenting your project planning that makes it easy for someone to replicate. 
            - Including 
                - my goals
                - a data dictionary, 
                - key findings and takeaways. 
            -  code well documented.
        
- Task out how you will work through the pipeline in as much detail as you need to keep on track.

    - Planning:
        - define the project itself in detail
        - define my ultimate goals of the project
        - how do I define succuess?
            - what is my deliverable?
                - how will I know I have succeeded?
        - how will I reach my goal?
            - how will i get from point a to point b
                - all the way down to when I give my presentation.
        - include my Data Dictionary
            - providing context of what is going to be used
            - explian my data
        - hypothesis 
            - these dont have to be your hard fast hypothesis.  
                - but I have looked at this data before when working on my story telling project. So I already have thought and mini hypothesis and those are what will be used in the planning stage.

    - Acquisition:
        - Create a path from my og data source to jupyter notebook so I can prepare and clean it.
            - In the acquire.py file (which has already been made)
                - store the proper imports to run all of the code
                - Create and store functions that needed to get the data from the telco chrun database.
                    - should return pandas data frame
        - determine the deliverable
            - the acquire file as stated up and the above bullet point.

    - Preparation:
        - In prepare.py store functions needed to prepare my data(important imports to run code)
            - The final function should be able to:
                - Split my data into train, validat, and test
                    - that can be explored, analyzed and visualized with ease.
                - Handle missing values
                - Handle incorrect data and outlier tht I may want to address.
                - Encode Variables
                - Create new features if wated
        - In my jupyter notebook
            - Clean my data
                - Explore missing values and document takeaways/action plans for handling them
                    - explain why I did what I did
                        - I dropped this column because ____________.
                        - I did this because of that
                    - is "missing" mean 0 or somehting else?
                    - should I replace any missing values with it most likely represents like mean, median, or mode?
                    - should i just remove the cariable column due to how much missing data there is?
                    - should I romove rows with missing data?
                - Explore data types and adapt types or data values as needed to have numeric represenations of each attribute
                    - are these data types correct?
                    - is it reading as an object when they are numeric?
                        - maybe one of the values has an object that is causing this issue?
            - Create any new features you want to use in your model. Some ideas you might want to explore after securing an MVP ( minimum viable product)
                - add a column to represent tenure but in years not months
                - single variables for or find other methods to merge variables representing the information from the following columns:
                    - phone_service and multiple_lines
                    - dependents and partner
                    - streaming_tv & streaming_movies
                    - online_security & online_backup
                - is there a minimum?

    - Exploration:
        - In my jupyter notebook:
            - Answers:
                - key questions
                - answer my hypothesis
                - determine the drivers of churn
                    - making sure to have at least 2 stat tests run
            - Makes sure to
                - Creating visualizations
                - Run statistical tests
                - Document findings based on stat tests ran
                    - the goal is to
                        - identify features that relate to churn
                        - indentify data integrity issues
                        - understand how my data works
                    - if there is a correlation
                        - assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation
            - Summarize my cnclusion, be able to answer question, and summarize takeaways and recomendations.

    - Modeling:
        - In my notebook:
            - Require to establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models.
                - Each step documented VERY well
            - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
            - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
            - Are there any variables that seem to provide limited to no additional information?
                - Yes?
                    - remove them
                - No?
                    - leave it be
            - Based on the evaluation choose the best model to test the data
            - Test the final model on my out of sample sata (the data set itself)
                - summarize , interpret, and document the results

    - Delivery:
        - Introduce myself
        - Introduce the project and goal
        - Summarize the findings (think of the Executive summary in a presenation)
        - The analysis I did to answer the questions and that lead to my findings. Relationships clearly visualized and takeaways are documented.
        - Finish with key takeaways, recommendations, and next steps and be prepared to answer questions from the data science team about your project.
        - Remember you have a time limit of 5 minutes for your presentation. Make sure you practice your notebook walkthrough keeping this time limit in mind; it will go by very quickly.

- Clearly state your starting hypotheses and add the testing of these to your task list.(going into it you already look at the data and did a project, state these thoguhts, they dont have to be formal hypothesis)
    - City, county, and zip code drive home pricing
    - Number of Bedrooms drives pricing
    - Number of bathrooms drive pricing
    - Homes with a central unit ac (id =1) have a higher price than others
    - The higher the square footage the higher the price


# Project Acquiring
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
    


# Project Exploration
 Plan -> Acquire -> Prepare -> **Explore** -> Model & Evaluate -> Deliver

1. pull my acquire
    `df = acquire.acquire_zillow()`
2. pull my prepare and prepare_telco
    `df = prepare.clean_zillow(df)`
    
    `df = prepare.focused_zillow(df)``
3. split my data
    `train, validate, test = prepare.split_focused_zillow(df)`
4. create correlation heat map
5. Create histograms for features being used
6. Run a stat test for each feature
7. Answering my questions:



# Project Modeling and Evaluation
 Plan -> Acquire -> Prepare -> Explore -> **Model & Evaluate** -> Deliver

1. Create prediction models with in sample data
2. Logit ran best with all 5 features
3. Run the Logit with out of sample



# Project Delivery
 Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> **Deliver**