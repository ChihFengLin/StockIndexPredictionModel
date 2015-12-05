%% Main Function
clc; clear; close all;

training = csvread('training.csv');
testing = csvread('testing.csv');
target = csvread('target.csv');
date_training = textread('date_training.csv', '%s', 'whitespace',',');
date_testing = textread('date_testing.csv', '%s', 'whitespace',',');


% Data normalization for training data(feature scaling between 0 to 1)
training = normalization(training);


[trained_classifier] = trainRBFClassifier(training, target, fold);