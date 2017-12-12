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
train['bought'].fillna((train['bought'].median()), inplace=True)
train['sold'].fillna((train['sold'].median()), inplace=True)
train.dropna(inplace=True)

xtrain = train.drop('return', axis=1)
ytrain = train[['return']]

#onehot encoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()

#type encoded
xtrain['type'] = le.fit_transform(xtrain['type'])

# pf_category encoded
xtrain['pf_category'] = le.fit_transform(xtrain['pf_category'])

# country code encoded
xtrain['country_code'] = le.fit_transform(xtrain['country_code'])

#currency encoded
xtrain['currency'] = le.fit_transform(xtrain['currency'])

#drop portfolio id
xtrain.drop('portfolio_id', axis=1, inplace=True)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [10])
xtrain = enc.fit_transform(xtrain)

#keras and it's library import
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#1st Layer
classifier.add(Dense(output_dim=6, input_dim=10))

#2nd layer
classifier.add(Dense(output_dim=6))

#output layer
classifier.add(Dense(output_dim=1))
classifier.add(Dense(output_dim=6, init = 'uniform', activation= 'linear', input_dim=11))

#2nd layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation= 'linear'))

#output layer
classifier.add(Dense(output_dim=1, init = 'uniform', activation= 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics= ['accuracy'])

classifier.get_config
#date conversion from minimum
min_date = xtrain['start_date'].min()
xtrain['start_date'] = xtrain['start_date'] - min_date
xtrain['creation_date'] = xtrain['creation_date'] - min_date
xtrain['sell_date'] = xtrain['sell_date'] - min_date

#fiting the ANN to the training set
classifier.fit(xtrain, ytrain, batch_size=10,  epochs=50)

#making the prediction and evaluating model
test = pd.read_csv("test.csv")
test = test[['pf_category', 'start_date', 'sold', 'country_code', 'euribor_rate', 'currency', 'bought', 'creation_date', 'sell_date', 'type']]

test['pf_category'] = le.fit_transform(test['pf_category'])
test['country_code'] = le.fit_transform(test['country_code'])
test['currency'] = le.fit_transform(test['currency'])
test['type'] = le.fit_transform(test['type'])

#date conversion from minimum
test['start_date'] = test['start_date'] - min_date
test['creation_date'] = test['creation_date'] - min_date
test['sell_date'] = test['sell_date'] - min_date

y_pred = classifier.predict(test)
y_pred = classifier.predict_proba(test)











#predicting the test results