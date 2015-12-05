import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

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


def scoreFeature(X, y):
	weights = np.zeros((1, X.shape[1]))

	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	ranking = []

	print("Feature Ranking:")
	for f in range(X.shape[1]):
		ranking.append(indices[f])
		print("%d. feature S%d (%f)" % (f + 1, indices[f] + 2, importances[indices[f]]))
	return ranking

def generateNewFeature(data, ranking):
	new_dataset = []
	for row in data:
		temp = []
		for rank in ranking[0:5]:      # Use top 5 important feature
			temp.append(row[rank])
		row.append(np.mean(temp))   # Calculate their mean
		row.append(np.std(temp))    # Calculate their std
		new_dataset.append(row)
	
	return new_dataset



if __name__ == '__main__':

	date_training, date_testing, training, testing, target = parseRawData(sys.argv[1])
	ranking = scoreFeature(np.matrix(training), target)
	new_training = generateNewFeature(training, ranking)
	new_testing = generateNewFeature(testing, ranking)
	
	# Save parsed data
	np.savetxt('./data/training.csv', new_training, delimiter=',')
	np.savetxt('./data/testing.csv', new_testing, delimiter=',')
	np.savetxt('./data/target.csv', target, delimiter=',')
	np.savetxt('./data/date_training.csv', date_training, delimiter=',', fmt="%s")
	np.savetxt('./data/date_testing.csv', date_testing, delimiter=',', fmt="%s")




