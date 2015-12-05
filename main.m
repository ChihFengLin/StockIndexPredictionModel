%% Main Function
clc; clear; close all;

% Read parsed data
training = csvread('./data/training.csv');
testing = csvread('./data/testing.csv');
target = csvread('./data/target.csv');
date_training = textread('./data/date_training.csv', '%s', 'whitespace',',');
date_testing = textread('./data/date_testing.csv', '%s', 'whitespace',',');

% Data normalization for training data(feature scaling between 0 to 1)
training = normalization(training);
testing = normalization(testing);

% Use cross-validation to train RBF kernel ridge regressor
% trainRBFClassifier will return [gamma lambda minRMSE stdOfRMSE] which gamma is the
% parameter for gaussian kernel and lambda is regularization parameter
fold = 10;
trained_classifier = trainRBFClassifier(training, target, fold);

% Apply trained classifier into out test data and make prediction
result = zeros(size(testing, 1), 1);
trained_gamma = trained_classifier(1, 1);
trained_lambda = trained_classifier(1, 2);
trained_sigma = sqrt(1/(2*trained_gamma));
KernelRegression = KernelRidgeRegression(['rbf'], training, trained_sigma, target, trained_lambda);
result(:, 1)=KernelPrediction(KernelRegression, testing);

% Generate output
answer = cell(size(result, 1) + 1, 2);
answer(1, :) = {'Date', 'Value'};
answer(2:size(result, 1) + 1, :) = [cellstr(date_testing) num2cell(result)];
cell2csv('prediction.csv', answer)
