# Stock market perdiction models base don Market Sentiment using python

# Objective: Stock market perdiction models based on Market Sentiment using python.

"""
Description:
1. Data Collection:
    Gather historical stock price data for top 100 stocks.
    Obtain market sentiment data from media, financial news articals or we can just use A Specialized Sentiment API (which is just a prebuilt sentiment analysis thing)
2. Sentiment Analysis
    Perform sentiment analysis on collective data to determine sentiment Positive negative or neutral (for this im thinking just 1 - positive, 0 is negative, 3 is neutral) if we want to make it ourselves we can or we can just use pretrained sentiment analysis models like google cloud natural language or whatever.
3. Feature Engineering:
    Combine the sentiment analysis results with stock price data.
    Create features from the sentiment data such as the total sentiment score for a day, sentiment ratio, or change in sentiment. 
4. Train-Test Split
    Split data into training and testing sets. The training set will be used to build the prediction model, and the testing set will be used to evaluate performance.
5. Model selection:
    We can basically choose a ml algorith for prediciton idk which one you want to use but we can do whatever
6. Model training:
    Train model.
7. Model Evaluation:
    Evaluate the model's performance useing MSE, MAE, RMSE (cringe stats stuff)
8. Model Deployment:
    Depoy model to make real time predictions or use it to forcast future stock prices based on sentiment. 

Disclaimer:
 stock prices are highly influenced by many other factors. while Sentiment analysis can provide insights it cant be a sole or 100% accurate perdictor of stock price perdiction. 

"""
