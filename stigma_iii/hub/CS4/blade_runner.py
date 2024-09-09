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
data = data.set_index('BBL')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#
data['BUILDING CLASS CATEGORY'] = data['BUILDING CLASS CATEGORY'].str.split().str[1]
data['TAX_CLASS_AT_PRESENT_P1'] = data['TAX CLASS AT PRESENT'].str[0].fillna('O')
data['TAX_CLASS_AT_PRESENT_P2'] = data['TAX CLASS AT PRESENT'].str[1].fillna('O')
data['BUILDING_CLASS_AT_PRESENT_P1'] = data['BUILDING CLASS AT PRESENT'].str[0].fillna('O')
data['BUILDING_CLASS_AT_PRESENT_P2'] = data['BUILDING CLASS AT PRESENT'].str[1].fillna('O')

data['LAND SQUARE FEET'] = pandas.to_numeric(data['LAND SQUARE FEET'], errors='coerce')
data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].fillna(data['LAND SQUARE FEET'].median())
data['GROSS SQUARE FEET'] = pandas.to_numeric(data['GROSS SQUARE FEET'], errors='coerce')
data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].fillna(data['GROSS SQUARE FEET'].median())
data.loc[data['YEAR BUILT'] == 0, 'YEAR BUILT'] = numpy.nan
data['YEAR BUILT'] = data['YEAR BUILT'].fillna(data['YEAR BUILT'].median())

data['BUILDING_CLASS_AT_TIME_OF_SALE_P1'] = data['BUILDING CLASS AT TIME OF SALE'].str[0].fillna('O')
data['BUILDING_CLASS_AT_TIME_OF_SALE_P2'] = data['BUILDING CLASS AT TIME OF SALE'].str[1].fillna('O')

data['SALE_DATE_YEAR'] = pandas.to_datetime(data['SALE DATE']).dt.year
data['SALE_DATE_MONTH'] = pandas.to_datetime(data['SALE DATE']).dt.month
data['SALE_DATE_DAY'] = pandas.to_datetime(data['SALE DATE']).dt.day

data['SALE PRICE'] = pandas.to_numeric(data['SALE PRICE'], errors='coerce')

#
removables = ['BLOCK', 'LOT', 'NEIGHBORHOOD', 'EASE-MENT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE',
              'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE',
              'SALE DATE']

target = 'SALE PRICE'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()
thresh = 10_000
data = data[data[target] > thresh]

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['TAX_CLASS_AT_PRESENT_P1', 'TAX_CLASS_AT_PRESENT_P2',
                'BUILDING_CLASS_AT_PRESENT_P1', 'BUILDING_CLASS_AT_PRESENT_P2',
                'BUILDING_CLASS_AT_TIME_OF_SALE_P1', 'BUILDING_CLASS_AT_TIME_OF_SALE_P2']

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

onehot = OneHotEncoder()
onehot_cols = ['BUILDING CLASS CATEGORY']
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
