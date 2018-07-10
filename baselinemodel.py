import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
class baseline_model():
    def __init__(self,input,output):
       model = Sequential()
       model.add(Dense(input, input_dim=input, kernel_initializer='normal', activation='relu'))
       model.add(Dense(output, kernel_initializer='normal', activation='softmax'))
    # Compile model
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       self.model = model
    def fit(self,X_train,y_train,epoch=10):
        self.model.fit(X_train, y_train, epochs=epoch, batch_size=200, verbose=2)
    def predict_class(X):
        i = self.model.predict_classes(X)
        return i[0]
