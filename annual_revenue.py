# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:27:27 2017

@author: suraj
"""

import pandas as pd
import sklearn

ttrain = pd.read_csv("train.csv")
train = ttrain[['portfolio_id', 'pf_category', 'start_date', 'sold', 'country_code', 'euribor_rate', 'currency', 'libor_rate', 'bought', 'creation_date', 'sell_date', 'type', 'return']]


train.drop(['libor_rate'], axis=1, inplace=True)
train['sold'].fillna((train['sold'].median()), inplace=True)
train.dropna(inplace=True)

xtrain = train.drop('return', axis=1)
ytrain = train[['return']]

#onehot encoder
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [10])
xtrain = enc.fit_transform(xtrain)

#keras and it's library import
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#1st Layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation= 'relu', input_dim=11))

#2nd layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation= 'relu'))

#output layer
classifier.add(Dense(output_dim=1, init = 'uniform', activation= 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

#fiting the ANN to the training set
classifier.fit(xtrain, ytrain, batch_size=10,  epochs=100)

#making the prediction and evaluating model

#predicting the test results