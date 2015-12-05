function [trained_classifier] = trainRBFClassifier(training, target, fold)

% k-fold cross validation
index = 1;
num_in_group = round(size(training, 1) / fold);
append_index = zeros(size(training, 1), 1);
for i = 1 : size(training, 1)
    if (mod(i,num_in_group) == 0)
        index = index + 1;
    end
    append_index(i, 1) = index;
end
train_append_index = [append_index training];
prediction_RMSE = zeros(fold, 1);
best_gamma = 0;
best_lambda = 0;
min_err = Inf;
std_val = 0;
trained_classifier = zeros(size(target, 2), 4);   % trained_classifier = [gamma lambda RMSE stdOfRMSE]

% rbf kernel ridge regression parameter (gamma, lambda) 
for gamma = [2^-17 2^-16 2^-15 2^-14 2^-13 2^-12 2^-11 2^10 2^-9 2^-8 2^-7 2^-6 2^-5 2^-4 2^-3]
    for lambda = [10 1 0.1 0.01 0.001 0.0001 0.00001]
        for i = 1 : fold
            % partition data for different fold
            idx_test = find(train_append_index(:, 1) == i);
            idx_train = find(train_append_index(:, 1) ~= i);
            trainData = training(idx_train, :); testData = training(idx_test,:);
            trainLabel = target(idx_train, 1); testLabel = target(idx_test, 1);

            % Kernelized Ridge Regression
            sigma = sqrt(1/(2*gamma));
            KernelRegression =KernelRidgeRegression(['rbf'], trainData, sigma, trainLabel, lambda);
            ypred=KernelPrediction(KernelRegression, testData);
            prediction_RMSE(i, 1) = sqrt(mean((ypred-testLabel).^2));
        end

        if mean(prediction_RMSE) < min_err
            min_err = mean(prediction_RMSE);
            std_val = std(prediction_RMSE);
            best_gamma = gamma;
            best_lambda = lambda;
        end
    end 
end
    
trained_classifier(1, 1) = best_gamma;
trained_classifier(1, 2) = best_lambda;
trained_classifier(1, 3) = min_err;
trained_classifier(1, 4) = std_val;

end


