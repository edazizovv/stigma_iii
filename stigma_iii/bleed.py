#
import json
import random


#
import numpy
import pandas
from sklearn.model_selection import train_test_split


import torch


#
from neura import WrappedNN
from hadok_gluth import rex


#
data = pandas.read_csv('./data/dataset.csv')
data = data.set_index('No')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#


#
removables = []

target = 'Y house price of unit area'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

X = data[x_factors].values.astype(dtype=float)
Y = data[target].values.astype(dtype=float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=rs)

#
param_grid = pandas.read_csv('./greed.csv')
for row in param_grid.iterrows():

    X_train_ = X_train.copy()
    Y_train_ = Y_train.copy()
    X_test_ = X_test.copy()
    Y_test_ = Y_test.copy()

    rex(X_train_, Y_train_, X_test_, Y_test_,
        dmr=row['dmr'], pre=row['pre'], fs=row['fs'],
        model_skeleton=WrappedNN, model_kwg=json.loads(row['model_kwg']),
        rs=rs)
