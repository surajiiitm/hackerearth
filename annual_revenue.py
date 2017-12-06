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
train.dropna(inplace=True)

xtrain = train.drop('return', axis=1)
ytrain = train[['return']]

