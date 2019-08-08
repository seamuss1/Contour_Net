import os, time, datetime
import cv2 as cv
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation



model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

