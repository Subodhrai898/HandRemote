import numpy as np
import dataprocess as dp
import perceptron as p
import baselinemodel as mt

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x , y , input , output = dp.process(X_train,y_train)
model1 = p.neuralPerceptron(input = input, output = output)
model1.fit(x,y)
X_test = dp.inputProcess(X_test)

x = np.zeros((1,784))
x[0] = X_test[1]
i = model1.predict_class(x)
print(i,y_test[1])
