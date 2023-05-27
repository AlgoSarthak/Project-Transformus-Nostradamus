import numpy as np
import matplotlib.pyplot as plt

# creating data
mean = np.array([5.0, 6.0])
cov = np.array([[1.0, 0.95], [0.95, 1.2]])
data = np.random.multivariate_normal(mean, cov, 8000)

data = np.hstack((np.ones((data.shape[0], 1)), data))

split_factor = 0.90
split = int(split_factor * data.shape[0])

print(split)

X_train = data[:split, :-1]
y_train = data[:split, -1].reshape((-1, 1))
X_test = data[split:, :-1]
y_test = data[split:, -1].reshape((-1, 1))

print(y_train.shape)

def hypothesis(X, theta):
	return np.dot(X, theta)

# function to compute gradient of error function w.r.t. theta


def gradient(X, y, theta):

	m = len(y)
	grad = (1/m)*np.dot(X.T, (hypothesis(X, theta) - y))	
	return grad

# function to compute the error for current values of theta


def cost(X, y, theta):
	m = len(y)
	predictions = hypothesis(X, theta)
	cost_func = (1/2*m)*np.sum((predictions - y)**2)
	return cost_func

# function to create a list containing mini-batches

def create_mini_batches(X, y, batch_size):
	mini_batches = []
	data = np.hstack((X, y))
	np.random.shuffle(data)
	n_minibatches = data.shape[0] // batch_size
	i = 0

	for i in range(n_minibatches):
		start = i*batch_size
		end = start + batch_size
		X_mini_batch = data[start:end, :-1]
		y_mini_batch = data[start:end, -1].reshape((-1, 1))
		mini_batches.append((X_mini_batch, y_mini_batch))

	if data.shape[0] % batch_size != 0:
		start = n_minibatches*batch_size
		X_mini_batch = data[start:, :-1]
		y_mini_batch = data[start:, -1].reshape((-1, 1))
		mini_batches.append((X_mini_batch, y_mini_batch))
		
	return mini_batches

# function to perform mini-batch gradient descent


def gradientDescent(X, y, learning_rate=0.001, batch_size=32):
	
	theta = np.zeros((2,1))
	error_list= []	
	mini_batches = create_mini_batches(X, y, batch_size)

	for X, y in mini_batches:
		# predictions = hypothesis(X, theta)
		# error = predictions - y
		grad = gradient(X, y, theta)
		cost_func = cost(X, y,theta)
		error_list.append(cost_func)
		theta -= learning_rate*grad

	return (theta, error_list)	
	
theta, error_list = gradientDescent(X_train, y_train)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


# predicting output for X_test
y_pred = hypothesis(X_test, theta)
plt.scatter(X_test[:, 1], y_test[:, ], marker='.')
plt.plot(X_test[:, 1], y_pred, color='orange')
plt.show()

error = hypothesis(X_train, theta) - y_train