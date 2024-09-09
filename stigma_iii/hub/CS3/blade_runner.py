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


#
removables = ['New_Price']

target = 'Price'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]


def treat_mileage(x):
    if 'km/kg' in x:
        return float(x[:x.index(' ')])
    elif 'kmpl' in x:
        return float(x[:x.index(' ')]) * 1.35
    else:
        return numpy.nan


data['Name'] = data['Name'].fillna(' ').apply(func=lambda x: x[:x.index(' ')])
data['Mileage'] = pandas.to_numeric(data['Mileage'].fillna(' ').apply(func=treat_mileage), errors='coerce')
data['Engine'] = pandas.to_numeric(data['Engine'].fillna(' ').apply(func=lambda x: x[:x.index(' ')]), errors='coerce')
data['Power'] = pandas.to_numeric(data['Power'].fillna(' ').apply(func=lambda x: x[:x.index(' ')]), errors='coerce')

data.loc[data['Power'].isna(), 'Power'] = data['Power'].median()
data.loc[data['Mileage'].isna(), 'Mileage'] = data['Mileage'].median()
data.loc[data['Engine'].isna(), 'Engine'] = data['Engine'].median()
data.loc[data['Seats'].isna(), 'Seats'] = data['Seats'].mode().values[0]

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type', ]

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

nochange_cols = [x for x in x_factors if x not in ordinal_cols]

X = numpy.concatenate((ordinal.transform(X=X[:, [x_factors.index(x) for x in ordinal_cols]]),
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
