# StockIndexPredictionModel

The goal of this project is to establish a prediction model for S&P index (S1) based on 9 stocks (S2~S10) whcih are part of the Nikkei Index. The data can be found in the data folder so-called stock_returns_base150.csv. We should build our model using the first 50 rows of the dataset. The remaining 50 rows of the dataset have values for S1 missing.

This problem is a regression problem and the model can be split into two parts: 
    
    STEP1. Estimate the importance of features (featureScoring.py)  
	STEP2. Training gaussian kernel ridge regression model (main.m)

## Feature Scoring (Feature Engineering)
*To execute this step, you need to install module "sklearn" and "numpy" and type the following command in the command line*

    python featureScoring.py ./data/stock_returns_base150.csv.

This feature scoring method is based on Random Forest to estimate the importance of each features. According to features' ranking, I choose the highest 5 features to calculate their mean and standard deviation. And then add them into the original feature set. I attempt to use important features to generate new features so that I can increase feature complexity.This might be helpful for establishing the prediction model in the second step.


## Training Gaussian Kernel Ridge Regression Model
*To execute this step, you need launch Matlab program and run main.m. It will generate the final prediction.csv file.*

For this problem, I choose kernelized ridge regression method with gaussian kernel which can effectively deal with non-linear dataset to achieve higher prediction accuracy. Among this model, there are two hyper-parameters: **gamma** and **lambda**. Gamma is the parameter for gaussian kernel and lambda is regularization parameter. In the main program, I implement n-fold cross-validation to tune the best combination of these two parameters and use RMSE to evaluate model performance.
    
    RMSE = sqrt(mean((predictedValue - trueValue).^2))
    
After training our gaussian kernel ridge regression model, I apply test data into this model and generate the final prediction.csv dataset.