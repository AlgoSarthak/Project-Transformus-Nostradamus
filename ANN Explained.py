import numpy as np
from keras.datasets import mnist
import random

#importing training and test data

(train_X, train_y), (test_X, test_y) = mnist.load_data()

#reshaping data from (28, 28) to (784, 1)

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

#zip acts like match-making, it pairs every x with y from index 1 to n
#Here we receive our zip into a list

train_data = list(zip(train_X, train_y))
test_data = list(zip(test_X, test_y))

#sigmoid function

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#Defining a class

class Network(object):
    def __init__(self,sizes): # sizes is a list containing the network. 
                              # eg : [784,128,10] means input =784 neurons,
                              #    1st hidden layer 128 neurons, output 10 neurons.
        self.sizes=sizes
        self.num_layers=4
        self.weights= [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases= [np.random.randn(x,1) for x in sizes[1:]]

# stop_zone 1. Comment out all the code below. Select all rows below. Click Ctrl + /.
# Include the show function given below above this comment area inside the class.
# Run this cell and then run the code with stop_zone 1 written below.
# After this testing, don't forget tto remove the comments. Same, select all, Ctrl+/.

    def forwardpropagation(self,a):
        for b,w in zip(self.biases, self.weights):
            a=sigmoid(np.dot(w, a) + b) # sig (w.a +b)
            # print(a.shape)
        return a

# stop_zone 2. Comment out all the code below. Don't comment out the __init__ method else you will get error.
# Remove comment from print(a.shape) line above. Run this cell. And run the code with stop_zone 2 written below.


    def backpropagation(self,x,y):
        
        # nothing to do in this 3 lines.
        y_t = np.zeros((len(y), 10))
        y_t[np.arange(len(y)), y] = 1
        y_t= y_t.T

        #nabla_b=dC/db and nabla_w=dC/dw. They are lists of shapes equal to that of bias and weights.
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        # initially, a0 = input.
        activation=x
        activation_list=[x]

        # step 1 : calculation of delta in last layer

        # write the same forward propagation code here but while doing so store the a's.
        for w,b in zip(self.weights,self.biases):
            activation= sigmoid(np.dot(w, activation) + b)
            activation_list.append(activation)

        delta= (activation_list[-1] - y_t)

        # step 2 : nabla_b and nabla_w relation with delta of last layer

        nabla_b[-1]= delta
        nabla_w[-1]= (delta)*(activation_list[-2].T)

        #print("{} {}".format(nabla_b[-1].shape,nabla_w[-1].shape) )
#stop_zone 3 : remove comment from the print statement just above and run the cell for stop_zone3.
# don't forget commenting out.

        # step 3 : calculation of delta for hidden layers
        
        for j in range(2,self.num_layers):
            sig_der = activation_list[-j]*(1-activation_list[-j])
            delta= np.dot(self.weights[1-j].T, delta)*(sig_der)

            # step 4 : nabla_b and nabla_w relation with delta of others layers
            nabla_b[-j]= delta
            nabla_w[-j]= np.dot(delta, activation_list[-1-j].T)
        
#stop_zone 4 : Run the cell for stop_zone 4.
        return (nabla_b,nabla_w)

    # the functions below are complete. If you are fine till stop_zone 4, you can run
    # this whole cell and train, test the data by running the last cell of the question.
    # You may need to wait for around 10 minutes to see the test predictions.
    def update_mini_batch(self,mini_batch,lr):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_b,delta_w= self.backpropagation(x,y)
            nabla_b=[nb+ db for nb,db in zip (nabla_b,delta_b)]
            nabla_w=[nw+dw for nw,dw in zip(nabla_w,delta_w)]

        self.weights=[w- lr*nw/len(mini_batch) for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-lr*nb/len(mini_batch) for b,nb in zip(self.biases,nabla_b)]


    def SGD(self, train_data,epochs,mini_batch_size, lr):
        n_train= len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+ mini_batch_size] for k in range(0,n_train,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,lr)

            self.predict(train_data)
            print("Epoch {0} completed.".format(i+1))
 
    def predict(self,test_data):
        test_results = [(np.argmax(self.forwardpropagation(x)),y) for x,y in test_data]
        # returns the index of that output neuron which has highest activation

        num= sum(int (x==y) for x,y in test_results)
        print ("{0}/{1} classified correctly.".format(num,len(test_data)))
         

net=Network([784,128,64,10])
net.SGD(train_data=train_data,epochs=10,mini_batch_size=20,lr=0.01)
print("Test data:")
net.predict(test_data)
