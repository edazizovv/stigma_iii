#
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
from hadok_gluth import nn_rex
from translator import translate_nn_kwargs


#
data = pandas.read_csv('./data/dataset.csv')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#
cities = data['ADDRESS'].str.split(',').str[1]
cities_chosen = cities.value_counts()[cities.value_counts() >= 1000].index
data['ADDRESS'] = cities.apply(func=lambda x: x if x in cities_chosen else 'OTHER')

#
removables = ['BHK_OR_RK']


target = 'TARGET(PRICE_IN_LACS)'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['POSTED_BY']

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

onehot = OneHotEncoder()
onehot_cols = ['ADDRESS']
onehot.fit(X=X[:, [x_factors.index(x) for x in onehot_cols]])

nochange_cols = [x for x in x_factors if x not in ordinal_cols + onehot_cols]

X = numpy.concatenate((ordinal.transform(X=X[:, [x_factors.index(x) for x in ordinal_cols]]),
                       onehot.transform(X=X[:, [x_factors.index(x) for x in onehot_cols]]).toarray(),
                       X[:, [x_factors.index(x) for x in nochange_cols]]), axis=1).astype(dtype=float)

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
