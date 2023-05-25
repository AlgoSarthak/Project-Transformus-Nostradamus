import numpy as np
from keras.datasets import mnist
import random

#importing training and test data

(train_X, train_y), (test_X, test_y) = mnist.load_data()

#reshaping data from 28 x 28 to 784 x 1

train_X = train_X.reshape(60000, 784, 1)
test_X = test_X.reshape(10000, 784, 1)

train_y = train_y.reshape(60000, 1)
test_y = test_y.reshape(10000, 1)

#normalizing data

train_X= train_X/255
test_X = test_X/255

#taking the first 10000 training images and first 1000 test data

train_X=train_X[:10000]         
train_y=train_y[:10000]
test_X=test_X[:1000]
test_y=test_y[:1000]

train_data = list(zip(train_X, train_y))
test_data = list(zip(test_X, test_y))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Network(object):
    def __init__(self,sizes):  

        self.sizes = sizes
        self.num_layers = 4
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    def show(self):
       print(self.num_layers)
       for bias in self.biases:
        print(bias.shape)
       for weight in self.weights:
        print(weight.shape)

net=Network([784,128,64,10])
net.show()