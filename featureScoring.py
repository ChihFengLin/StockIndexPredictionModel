import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

def parseRawData(fileName):

	fd = open(fileName, 'rb')
	date_training = []
	date_testing = []
	training = []
	testing = []
	target = []

	try:
	    reader = csv.reader(fd)  
	    for row in reader:

	    	if row[0] == '':   # Skip empty line
	    		break   
	        
	    	if row[0] == 'date':   # Skip header line
	    		continue

	        if row[1] == '':   # Check start of testing data
	        	date_testing.append(row[0])
	        	temp = []
	        	for number in row[2:]:
	        		temp.append(float(number))
	        	testing.append(temp)
	        	continue

	        date_training.append(row[0])
	        target.append(float(row[1]))
	        temp = []
	        for number in row[2:]:
	        	temp.append(float(number))
	        training.append(temp)

	finally:
	    fd.close()

	return date_training, date_testing, training, testing, target


if __name__ == '__main__':
	date_training, date_testing, training, testing, target  = parseRawData('stock_returns_base150.csv')
	print target

#training = genfromtxt('stock_returns_base150.csv', delimiter=',')

#print training

#labels = genfromtxt('labels.csv', delimiter=',')
#labels = labels.transpose()
#X = training

weights = np.zeros((labels.shape[0], X.shape[1]))

for classifier_no in range(labels.shape[0]):
	
	y = labels[classifier_no][0:]

	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	indices = np.argsort(importances)

	for f in range(X.shape[1]):
		weight = f / float(X.shape[1] - 1)
		weights[classifier_no][indices[f]-1] = weight
    	#print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#np.savetxt('weights.csv', weights, delimiter=',')
