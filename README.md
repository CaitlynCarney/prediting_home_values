# Project Information

## Scenario
- target is your taxamount
    - do not use landtaxdollarcnt
    - taxvaluedollarcnt
​
- Goal is to predict a homes value
    - but what are you defining as the homes value
        - what the county assesses the county at?
​
- You are a junior data scientist on the Zillow data science team and recieve the following email in your inbox:
    - We want to be able to predict the values of single unit properties that the tax district assesses using the property data from those with a `transaction` during the "hot months" 
        - **May-August, 2017.**
- We also need some additional information outside of the model.
    - Zach lost the email that told us where these properties were located. Ugh, Zach :-/. 
        - Because property taxes are assessed at the county level, we would like to know what **states and counties these are located in**.
- We'd also like to know the **distribution of tax rates for each county**.
    - a histogram would work with this
        - 1 hist per county
        - 1 clean layered histogram
            - ONLY IF IT IS CLEAN ENOUGH
    - The data should have the tax amounts and tax value of the home, so it shouldn't be too hard to calculate. 
    - Please include in your report to us the distribution of tax rates for each county so that we can see how much they vary within the properties in the county and the rates the bulk of the properties sit around.
        - Note that this is separate from the model you will build, because if you use tax amount in your model, you would be using a future data point to predict a future data point, and that is cheating! 
            - In other words, for prediction purposes, we won't know tax amount until we know tax value.
            
## Audience
- Your customer/end user is the **Zillow data science team**. 
- In your deliverables, be sure to re-state your goals, as if you were delivering this to Zillow. 
    - They have asked for something from you, and you are basically communicating in a more concise way, and very clearly, the goals as you understand them and as you have taken and acted upon them through your research.
    
# Deliverables

- delivering to zillow data science team
    - use correct technical language for that group
- Remember that you are communicating to the **Zillow team**, not to your instructors. So, what does the team expect to receive from you?

See the Pipeline guidance below for more information on expectations within these deliverables.

1. A report in the form of a presentation, verbal supported by slides.
    - The report/presentation slides should summarize your findings about the drivers of the single unit property values.
        - This will come from the analysis you do during the exploration phase of the pipeline. 
            - In the report, you should have visualizations that support your main points.
    - **The presentation should be no longer than 5 minutes.**
    - should follow same guidelines of have agenda, exec summary, etc.

2. A github repository containing your work.
    - This repository should contain 
        - one clearly labeled final Jupyter Notebook that walks through the pipeline
            - but, if you wish, you may split your work among 2 notebooks, one for exploration and one for modeling.
        - In exploration, you should perform your analysis including: 
            - the use of at least two statistical tests 
            - visualizations documenting hypotheses 
            - takeaways. 
        - In modeling, you should 
            - establish a baseline that you attempt to beat with various algorithms and/or hyperparameters. 
            - Evaluate your model by computing the metrics and comparing.
    - Make sure your notebook answers all the questions posed in the email from the Zillow data science team.
    - The repository should also contain the 
        - .py files necessary to reproduce your work, 
            - acquire.py
            - prepare.py
            - warngle.py
                - if you choose to combine your acuire and prepare into one
        - your work must be reproducible by someone with their own env.py file.
    - As with every project you do, you should have 
        - an excellent README.md file documenting your project planning with instructions on how someone could clone and reproduce your project on their own machine. 
        - Include at least 
            - your goals for the project, 
            - a data dictionary, 
            - key findings and takeaways. 
        - Your code should be well documented.
    - No more than 2 Notebooks
        - want to see the exploration
            - not really all the acquire and prep you want these to be final functions that are puilled in from a .py file
            
## What I will need for the project
- you will need to reference the 
    - properties_2017 table 
    - predictions_2017 table
    - can reference more but you dont need them

- For the first iteration of your model, 
    - use only
        - square feet of the home, 
        - number of bedrooms, 
        - number of bathrooms 
    - to estimate the property's 
        - assessed value, 
        - taxvaluedollarcnt. 
    - You can expand this to other fields after you have completed an mvp (minimally viable product).

- You will need to figure out which field gives you the annual tax amount for the property in order to calculate the tax rate.
    - Using the property's assessed value (taxvaluedollarcnt) and the amount they pay each year (field name) to compute tax rate.

- You will want to read and re-read the requirements given by your stakeholders to be sure you are meeting all of their needs and representing it in your data, report, and model.

- You will want to do some data validation or QA (quality assurance) to be sure the data you gather is what you think it is.

- You will want to make sure you are using the best fields to represent square feet of home, number of bedrooms, and number of bathrooms. 
    - "Best" meaning the most accurate and available information. 
    - Here you will need to do some data investigation in the database and use your domain expertise to make some judgement calls.