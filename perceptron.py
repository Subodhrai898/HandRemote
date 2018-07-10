import numpy as np


class neuralPerceptron(object):

    def __init__(self,learning_rate = .7,epoch = 10,input=784,second_layer = 64,output=10):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.n_x = input
        self.n_h = second_layer
        self.W1 = np.random.randn(self.n_h,self.n_x)
        self.b1 = np.zeros((self.n_h,1))
        self.W2 = np.random.randn(output,self.n_h)
        self.b2 = np.zeros((output,1))

    def compute_multiclass_loss(self,Y,Y_hat):
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1/m) * L_sum

        return L
    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        return s

    def fit(self,X_train,Y_train):
        m = X_train.shape[0]
        X = X_train.T
        Y = Y_train.T

        Z1 = []
        A1 = []
        Z2 = []
        A2 = []
        for i in range(self.epoch):


          Z1 = np.matmul(self.W1,X) + self.b1
          A1 = self.sigmoid(Z1)
          Z2 = np.matmul(self.W2,A1) + self.b2
          A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

          cost = self.compute_multiclass_loss(Y, A2)

          dZ2 = A2-Y
          dW2 = (1./m) * np.matmul(dZ2, A1.T)
          db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

          dA1 = np.matmul(self.W2.T, dZ2)
          dZ1 = dA1 * self.sigmoid(Z1) * (1 - self.sigmoid(Z1))
          dW1 = (1./m) * np.matmul(dZ1, X.T)
          db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

          self.W2 = self.W2 - self.learning_rate * dW2
          self.b2 = self.b2 - self.learning_rate * db2
          self.W1 = self.W1 - self.learning_rate * dW1
          self.b1 = self.b1 - self.learning_rate * db1

          if (i % 100 == 0):
                print("Epoch", i, "cost: ", cost)
        print("final cost :",cost)


    def predict_class(self,X):
        X = X.T
        Z1 = np.matmul(self.W1,X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.W2,A1) + self.b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
        A2 = A2>0.5
        a = A2.flatten()
        print(a)
        i = np.where(a==True)
        return i[0][0]
