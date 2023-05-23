import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#import training samples
x, y = make_regression(n_samples=100, n_features=1, noise=10)

def gradient_descent(x, y, learning_rate, num_iterations):

    #Initialize parameter theta(weight matrix)
    theta = np.zeros(1)

    m = len(y)
    cost_progress = []

    for i in range(num_iterations):

        #Calculate predictions
        predictions = np.dot(x, theta)

        #Calculate error
        error = predictions - y

        #Calculate gradient 
        gradient = (1/m)*np.dot(x.T, error)

        #Apply gradient descent algorithm
        theta -= learning_rate*gradient

        #Calculate cost function
        cost = (1/2*m)*np.sum(error**2)

        cost_progress.append(cost)
    
    return theta, cost_progress

learning_rate = 0.01
num_iterations = 1000

theta, cost_progress = gradient_descent(x, y, learning_rate, num_iterations)

# Plot the cost function over iterations
plt.plot(range(num_iterations), cost_progress)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

# Plot the best fit line
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x, np.dot(x, theta), color='blue', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Fit Line')
plt.legend()
plt.show()