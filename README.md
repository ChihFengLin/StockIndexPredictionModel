# StockIndexPredictionModel

The goal of this project is to establish a prediction model for S&P index (S1) based on 9 stocks (S2~S10) whcih are part of the Nikkei Index. The data can be found in the data folder so-called stock_returns_base150.csv. We should build our model using the first 50 rows of the dataset. The remaining 50 rows of the dataset have values for S1 missing.

This problem is a regression problem and the model can be splited into two parts: 
    
    1. Estimate the importance of features (featureScoring.py)  
	2. Training gaussian kernel ridge regression model (main.m)

## Feature Scoring
*To execute this step, you need to install module "sklearn" and "numpy" and type the following command in the command line*

`python featureScoring.py ./data/stock_returns_base150.csv.`

This is feature scoring method is based on Random Forest to estimate the importance of each features. According to features' ranking, I choose the highest 5 features to calculate their mean and standard deviation. And then add them into the original feature set. I attempt to use important features to generate new features so that I can increase feature complexity.This might be helpful for establishing the prediction model in the second step.


## Training Gaussian Kernel Ridge Regression Model