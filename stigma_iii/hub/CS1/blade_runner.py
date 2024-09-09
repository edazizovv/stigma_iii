#
import json
import random

#
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


import torch
from torch import nn


#
from neura import WrappedNN
from hadok_gluth import nn_rex
from translator import translate_nn_kwargs


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

for i, row in param_grid.iterrows():
    print('RUN {0} / {1}'.format(i+1, param_grid.shape[0]))
    X_train_ = X_train.copy()
    Y_train_ = Y_train.copy()
    X_test_ = X_test.copy()
    Y_test_ = Y_test.copy()

    nn_rex(row['ex'], X_train_, Y_train_, X_test_, Y_test_,
           dmr=row['dmr'], pre=row['pre'], fs=row['fs'],
           model_skeleton=WrappedNN, model_kwg=translate_nn_kwargs(json.loads(row['model_kwg'])),
           rs=rs)
