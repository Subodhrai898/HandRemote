import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
class CNN(object):
    def __init__(self,input=10,output = 10):
        seed = 7
        np.random.seed(seed)
        model = Sequential()
	    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))
	    model.add(Dropout(0.2))
	    model.add(Flatten())
	    model.add(Dense(128, activation='relu'))
	    model.add(Dense(output, activation='softmax'))
	# Compile model
	    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	    self.model =  model

    def fit(self,X_train,y_train,epoch=10):
        self.model.fit(X_train, y_train, epochs= epoch, batch_size=200, verbose=2)
    def predict_class(X):
        i = self.model.predict_classes(X)
        return i[0]
