import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#sklearn.datasets contain sample examples which can we imported to in your code
#make_regression contain sets of examples to train a linear regression model

x, y = make_regression(n_samples=5, n_features=1, noise=10)

#Here .shape allows you to print the rows and colums of the example x

learning_rate = 0.01

#Here x is (5,1) while y is (5,)
###whenever (m,) is there then it can be imagined as a list with m elements in a single row

theta = np.zeros(1)         #theta is of type (1,)

m = len(y)

predictions = np.dot(x, theta)    #dot product of x(5,1) and theta(1,) gives predictions(5,)
#print(y, predictions)

error = predictions - y      #y(5,) and predictions(5,) gives error(5,)
#in order to remove negative sign from gradient we calculate error by -(y - predictions)

gradient = (1/m)*np.dot(x.T, error)     #dot product of x.T(x transpose)(1,5) and error(5,) gives gradient(1,)

print(gradient)
#print(x,y)
#print(x.T.shape, error.shape, y.shape, theta.shape)

theta =- learning_rate*gradient


#Cost function
print(error, (error)**2)

cost = (1/2*m)*np.sum(error**2)  #since value of the cost function is a single scalar value we have to add the values of error squared matrix
print(cost)