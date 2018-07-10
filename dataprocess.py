#data processing
import numpy
from keras.utils import np_utils

def process(X_train,y_train):
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_train = X_train / 255

    y_train = np_utils.to_categorical(y_train)

    num_classes = y_train.shape[1]

    return X_train,y_train,num_pixels,num_classes
def inputProcess(X):
    num_pixels = X.shape[1] * X.shape[2]
    X = X.reshape(X.shape[0], num_pixels).astype('float32')
    X = X / 255
    return X
