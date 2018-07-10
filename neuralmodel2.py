from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
class Model2(object):
    def __init__(self,input=784,output=10):
        model = Sequential()
        model.add(Dense(512, input_shape=(input,)))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(output))
        model.add(Activation('softmax'))
        # compiling the sequential model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        self.model = model


    def fit(self,X_train,y_train,epoch=10):

           self.model.fit(X_train, y_train, epochs=epoch, batch_size=200, verbose=2)
    def predict_class(X):
         i = self.model.predict_classes(X)
         return i[0]
