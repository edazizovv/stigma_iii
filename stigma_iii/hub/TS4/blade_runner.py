#
import json
import random

#
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

import torch
from torch import nn


#
from neura import WrappedNN
from hadok_gluth import nn_rex_ts
from translator import translate_nn_kwargs


#
pyplot.style.use('dark_background')


data = pandas.read_csv('./data/dataset.csv')
data = data.set_index('date')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#


#
removables = []

target = 'population'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors]

X = data[[target] + x_factors].values
Y = data[target].values

thresh = 0.5
start = 0
mid = int(X.shape[0] * thresh)
end = -1
X_train, X_test, Y_train, Y_test = X[start:mid, :], X[mid:end, :], Y[start:mid], Y[mid:end]
#
param_grid = pandas.read_csv('./greed.csv')

for i, row in param_grid.iterrows():
    print('RUN {0} / {1}'.format(i+1, param_grid.shape[0]))
    X_train_ = X_train.copy()
    Y_train_ = Y_train.copy()
    X_test_ = X_test.copy()
    Y_test_ = Y_test.copy()

    if 'LSTM' in json.loads(row['model_kwg']):
        lagger = 'SEQUENTIAL'
    else:
        lagger = 'SIMPLE'

    nn_rex_ts(row['ex'], X_train_, Y_train_, X_test_, Y_test_,
              dmr=row['dmr'], pre=row['pre'], fs=row['fs'],
              model_skeleton=WrappedNN, model_kwg=translate_nn_kwargs(json.loads(row['model_kwg'])),
              rs=rs, diff_type='PCT', lagger=lagger, window=row['lag'])
