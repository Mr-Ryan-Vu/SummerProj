#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import sys
import matplotlib.pyplot as plt # plotting
plt.style.use('ggplot')
import numpy as np #linear alg
import os # accessing directory structure
import pandas as pd #CSV data 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import nltk
nltk.download('vader_lexicon')

import warnings
warnings. filterwarnings("ignore")


# In[2]:


df1 = pd.read_csv('AppleFinalData.csv')


# In[3]:


df2 = pd.read_csv('AppleNewsStock.csv')


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sid = SentimentIntensityAnalyzer()

neg, neu, pos, compound = [], [], [], []

def sentiment(df):
    for i in range(len(df)):
        sen = unicodedata.normalize('NFKD', train.iloc[i]['News'])
        ss = sid.polarity_scores(sen)
        neg.append(ss['neg'])
        neu.append(ss['neu'])
        pos.append(ss['pos'])
        compound.append(ss['compound'])
    df['neg'] = neg
    df['neu'] = neu
    df['pos'] = pos
    df['compound'] = compound
    
    """In summary, this code calculates sentiment scores 
    (negative, neutral, positive, and compound) for each piece of text in the 
    "News" column of the DataFrame df using the VADER sentiment analyzer and 
    adds these scores as new columns to the DataFrame."""


# In[ ]:


def if_news(column):
    if column ==0:
        return 0
    else:
        return 1


# In[ ]:


df1['if_news'] = df1['compound'].apply(if_news)

df_weekly = df1[['Date','Close', 'compound', 'neg', 'pos', 'if_news']]
df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])
df_weekly.set_index('Date',inplace=True)
#creates a new column in df1 called 'if_news'it uses apply to apply if_news into each compound 
#column and they can be put into binary values where 0 indicates


# In[ ]:


def take_last(array_like):
    return array_like[-1]


# In[ ]:


output = df_weekly.resample('W', loffset=pd.Timedelta(days=-6)).agg({
    'Close': 'last',
    'compound': 'mean',
    'neg': 'mean',
    'pos': 'mean',
    'if_news': 'max'
})
# calculates mean for the weekly sentiment score for neg and pos, 
#then close means where the value 
# of the stock closed. 


# In[ ]:


output.head(10)


# In[ ]:


"""
Perform k-fold cross-validation on the data

Parameters
----------
df: DataFrame containing the data with date as the index
X: DataFrame containing the independent variable(s) (i.e., features, predictors)
y: DataFrame containing the dependent variable (i.e., the target)
k: Number of folds for cross-validation (default is 5)

Optional
--------
seed: Integer used as the seed for a random number generator
      You don't necessarily NEED to use a random number generator, but if you do, please use the default value for seed

Returns
-------
List of tuples: Each tuple contains (X_train, X_test, y_train, y_test) for one fold
"""


# In[ ]:


def k_fold_split(df, X, y, k=5, seed=42):

    # Create the random number generator
    rng = np.random.RandomState(seed)
    
    # Create k-fold cross-validator
    kfold = KFold(n_splits=k, shuffle=True, random_state=rng)
    
    # Initialize list to store the train-test splits
    splits = []
    
    for train_idx, test_idx in kfold.split(df):
        # Split the data into train and test sets based on the index
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Add the current fold split to the list
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits


# In[ ]:


def pd2ndarray( dfList ):
    """
    For each DataFrame in the list dfList, prepare the ndarray needed by the sklearn model
    
    Parameters
    ----------
    dfList: List of DataFrames
    
    Returns
    --------
    ndList: a list of ndarrays
    """
    
    # Create the function body according to the spec
    ndList = []
    for i in dfList:
        ndList.append(i.values)
    
    # Change the return statement as appropriate
    return ndList

