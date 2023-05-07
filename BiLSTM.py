# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:44:43 2023

@author: Jola
"""


from keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers import  Dense, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from matplotlib import pyplot
import os
import numpy as np

n_unique_words = 10000 # cut texts after this number of words
maxlen = 200
batch_size = 128 

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=n_unique_words)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test) 

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

history=model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=[x_test, y_test])
print(history.history['loss'])
print(history.history['accuracy'])

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['accuracy'])
pyplot.title('model loss vs accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['loss', 'accuracy'], loc='upper right')
pyplot.show()






