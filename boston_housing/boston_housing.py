"""
Loading the boston dataset and examining its target (label) distribution.
"""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
import sklearn
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
import collections

def load_data():
	'''Load the Boston dataset.'''

	boston = datasets.load_boston()
	return boston


def explore_city_data(city_data):
	'''Calculate the Boston housing statistics.'''

	# Get the labels and features from the housing data
	housing_prices = city_data.target
	housing_features = city_data.data

	
	###################################
	### Step 1. YOUR CODE GOES HERE ###
	###################################
	
	# Please calculate the following values using the Numpy library
	# Size of data?
	print "Size of the data is: %d" %len(housing_prices)
	
	# Number of features?
	print "Number of features for the data is: %d" %len(housing_features[0])
	
	# Minimum value?
	print "Minimum Housing price in Boston is: %0.2f" %min(housing_prices)
	
	# Maximum Value?
	print "Maximum Housing price in Boston is: %0.2f" %max(housing_prices)
	
	# Frequency Distribution ?
	counter =collections.Counter(housing_prices)

	# Calculate mean?
	print "Mean Housing price in Boston is: %0.2f" %np.mean(housing_prices)
	
	# Calculate median?
	print "Median Housing price in Boston is: %0.2f" %np.median(housing_prices)
	
	# Calculate standard deviation?
	print "Standard deviation of  Housing price in Boston is: %0.2f" %np.std(housing_prices)
	
	pl.figure()
	pl.hist(housing_prices, bins=20)
	pl.title("Housing Prices Histogram")
	pl.xlabel("Value")
	pl.ylabel("Frequency")
	pl.show()

def performance_metric(label, prediction):
	'''Calculate and return the appropriate performance metric.'''

	###################################
	### Step 2. YOUR CODE GOES HERE ###
	###################################

	# http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
	return metrics.mean_squared_error(label, prediction)
	


def split_data(city_data):
	'''Randomly shuffle the sample set. Divide it into training and testing set.'''

	# Get the features and labels from the Boston housing data
	X, y = city_data.data, city_data.target

	###################################
	### Step 3. YOUR CODE GOES HERE ###
	###################################

	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.30, random_state=42)

	return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
	'''Calculate the performance of the model after a set of training data.'''

	# We will vary the training set size so that we have 50 different sizes
	sizes = np.linspace(1, len(X_train), 50)
	train_err = np.zeros(len(sizes))
	test_err = np.zeros(len(sizes))

	print "Decision Tree with Max Depth: "
	print depth

	for i, s in enumerate(sizes):

		# Create and fit the decision tree regressor model
		regressor = DecisionTreeRegressor(max_depth=depth)
		regressor.fit(X_train[:s], y_train[:s])

		# Find the performance on the training and testing set
		train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
		test_err[i] = performance_metric(y_test, regressor.predict(X_test))


	# Plot learning curve graph
	learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
	'''Plot training and test error as a function of the training size.'''

	pl.figure()
	pl.title('Decision Trees: Performance vs Training Size')
	pl.plot(sizes, test_err, lw=2, label = 'test error')
	pl.plot(sizes, train_err, lw=2, label = 'training error')
	pl.legend()
	pl.xlabel('Training Size')
	pl.ylabel('Error')
	pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
	'''Calculate the performance of the model as model complexity increases.'''

	print "Model Complexity: "

	# We will vary the depth of decision trees from 2 to 25
	max_depth = np.arange(1, 25)
	train_err = np.zeros(len(max_depth))
	test_err = np.zeros(len(max_depth))

	for i, d in enumerate(max_depth):
		# Setup a Decision Tree Regressor so that it learns a tree with depth d
		regressor = DecisionTreeRegressor(max_depth=d)

		# Fit the learner to the training data
		regressor.fit(X_train, y_train)

		# Find the performance on the training set
		train_err[i] = performance_metric(y_train, regressor.predict(X_train))

		# Find the performance on the testing set
		test_err[i] = performance_metric(y_test, regressor.predict(X_test))

	# Plot the model complexity graph
	model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
	'''Plot training and test error as a function of the depth of the decision tree learn.'''

	pl.figure()
	pl.title('Decision Trees: Performance vs Max Depth')
	pl.plot(max_depth, test_err, lw=2, label = 'test error')
	pl.plot(max_depth, train_err, lw=2, label = 'training error')
	pl.legend()
	pl.xlabel('Max Depth')
	pl.ylabel('Error')
	pl.show()


def fit_predict_model(city_data):
	'''Find and tune the optimal model. Make a prediction on housing data.'''

	# Get the features and labels from the Boston housing data
	X, y = city_data.data, city_data.target
	print X
	# Setup a Decision Tree Regressor
	regressor = DecisionTreeRegressor()

	parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

	###################################
	### Step 4. YOUR CODE GOES HERE ###
	###################################
	
	# 1. Find the best performance metric
	# should be the same as your performance_metric procedure
	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

	reg = GridSearchCV(regressor, parameters,scoring=make_scorer(metrics.mean_squared_error,greater_is_better=False))
	print reg.fit(X, y)

	# 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
	# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

	depth_values= list()
	for i in xrange(101):
		reg.fit(X,y)
		depth_values.append(int(reg.best_params_['max_depth']))

	print "Best model parameter:  " + str(np.median(depth_values))
	# Fit the learner to the training data

    # Use the model to predict the output of a particular sample
	regressor = DecisionTreeRegressor(max_depth=np.median(depth_values))
	print "Final Model: "
	print regressor

	regressor.fit(X, y)
	
	x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
	y = regressor.predict(x)
	print "House: " + str(x)
	print "Prediction: " + str(y)


def main():
	'''Analyze the Boston housing data. Evaluate and validate the
	performanance of a Decision Tree regressor on the Boston data.
	Fine tune the model to make prediction on unseen data.'''

	# Load data
	city_data = load_data()
	
	# Explore the data
	explore_city_data(city_data)

	# Training/Test dataset split
	X_train, y_train, X_test, y_test = split_data(city_data)
	
	# Learning Curve Graphs
	max_depths = [1,2,3,4,5,6,7,8,9,10]
	for max_depth in max_depths:
		learning_curve(max_depth, X_train, y_train, X_test, y_test)

	# Model Complexity Graph
	model_complexity(X_train, y_train, X_test, y_test)
	
	# Tune and predict Model
	fit_predict_model(city_data)

main()