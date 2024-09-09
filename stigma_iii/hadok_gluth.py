#


#
import numpy
import pandas
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch

#
from tests import check_up
from recurrent import prepare_inputs, simulate_lags


#
def nn_rex(ex, X_train, Y_train, X_test, Y_test, dmr, pre, fs, model_skeleton, model_kwg, rs):

    if dmr == 'NO':
        X_train_ = X_train
        X_test_ = X_test
    elif dmr == 'PCA_MLE':
        # proj_rate = 0.50  # 0.75   0.5   0.25  'mle'
        # njv = int(X_train_.shape[1] * proj_rate)
        njv = 'mle'
        # njv = 0.75  # 0.75  0.5  0.25
        projector = PCA(n_components=njv, svd_solver='full', random_state=rs)
        projector.fit(X=X_train)
        X_train_ = projector.transform(X_train)
        X_test_ = projector.transform(X_test)
    else:
        raise ValueError("Invalid REX parameter `dmr` value passed: {0}".format(dmr))

    if pre == 'NO':
        pass
    elif pre == 'STD_SCAL':
        scaler = StandardScaler()
        scaler.fit(X=X_train)
        X_train_ = scaler.transform(X_train_)
        X_test_ = scaler.transform(X_test_)
    else:
        raise ValueError("Invalid REX parameter `pre` value passed: {0}".format(pre))

    if fs == 'NO':
        pass
    elif fs == 'KT_05':
        alpha = 0.05
        values = numpy.array([kendalltau(x=X_train_[:, j], y=Y_train)[1] for j in range(X_train_.shape[1])])
        fs_mask = values <= alpha
        X_train_ = X_train_[:, fs_mask]
        X_test_ = X_test_[:, fs_mask]
    else:
        raise ValueError("Invalid REX parameter `fs` value passed: {0}".format(fs))

    X_train_ = torch.tensor(X_train_, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float)
    X_test_ = torch.tensor(X_test_, dtype=torch.float)
    Y_test = torch.tensor(Y_test, dtype=torch.float)

    model = model_skeleton(**model_kwg)

    model.fit(X_train=X_train_, Y_train=Y_train.reshape(-1, 1),
              X_val=X_test_, Y_val=Y_test.reshape(-1, 1))

    y_hat_train = model.predict(X=X_train_)
    y_hat_test = model.predict(X=X_test_)

    results_train = check_up(Y_train.numpy(), y_hat_train.flatten(), None, X_train_)
    results_test = check_up(Y_test.numpy(), y_hat_test.flatten(), None, X_test_)

    results_train['sample'] = 'train'
    results_test['sample'] = 'test'

    results_train = pandas.DataFrame(pandas.Series(results_train))
    results_test = pandas.DataFrame(pandas.Series(results_test))

    results_train = results_train.T
    results_test = results_test.T

    results_train.index = [ex]
    results_test.index = [ex]

    results_train.to_csv('./reported.csv', mode='a', header=False)
    results_test.to_csv('./reported.csv', mode='a', header=False)


def nn_rex_ts(ex, X_train, Y_train, X_test, Y_test, dmr, pre, fs, model_skeleton, model_kwg, rs,
              diff_type, lagger, window):

    _X_train = X_train.copy()
    _X_test = X_test.copy()
    _Y_train = Y_train.copy()
    _Y_test = Y_test.copy()

    if diff_type == 'PCT':
        X_train = pandas.DataFrame(X_train).pct_change().values[1:]
        Y_train = pandas.DataFrame(Y_train).pct_change().values[1:]
        X_test = pandas.DataFrame(X_test).pct_change().values[1:]
        Y_test = pandas.DataFrame(Y_test).pct_change().values[1:]

    elif diff_type == 'DIFF':
        X_train = pandas.DataFrame(X_train).diff().values[1:]
        Y_train = pandas.DataFrame(Y_train).diff().values[1:]
        X_test = pandas.DataFrame(X_test).diff().values[1:]
        Y_test = pandas.DataFrame(Y_test).diff().values[1:]
    else:
        raise ValueError("Invalid REX parameter `diff_type` value passed: {0}".format(diff_type))

    for j in range(X_train.shape[1]):
        X_train[~numpy.isfinite(X_train[:, j]), j] = numpy.ma.masked_invalid(X_train[:, j]).max()
        X_test[~numpy.isfinite(X_test[:, j]), j] = numpy.ma.masked_invalid(X_test[:, j]).max()
    Y_train[~numpy.isfinite(Y_train)] = numpy.ma.masked_invalid(Y_train).max()
    Y_test[~numpy.isfinite(Y_test)] = numpy.ma.masked_invalid(Y_test).max()

    if dmr == 'NO':
        X_train_ = X_train
        X_test_ = X_test
    elif dmr == 'PCA_MLE':
        # proj_rate = 0.50  # 0.75   0.5   0.25  'mle'
        # njv = int(X_train_.shape[1] * proj_rate)
        njv = 'mle'
        # njv = 0.75  # 0.75  0.5  0.25
        projector = PCA(n_components=njv, svd_solver='full', random_state=rs)
        projector.fit(X=X_train)
        X_train_ = projector.transform(X_train)
        X_test_ = projector.transform(X_test)
    else:
        raise ValueError("Invalid REX parameter `dmr` value passed: {0}".format(dmr))

    if pre == 'NO':
        pass
    elif pre == 'STD_SCAL':
        scaler = StandardScaler()
        scaler.fit(X=X_train)
        X_train_ = scaler.transform(X_train_)
        X_test_ = scaler.transform(X_test_)
    else:
        raise ValueError("Invalid REX parameter `pre` value passed: {0}".format(pre))

    if fs == 'NO':
        pass
    elif fs == 'KT_05':
        alpha = 0.05
        values = numpy.array([kendalltau(x=X_train_[:, j], y=Y_train)[1] for j in range(X_train_.shape[1])])
        fs_mask = values <= alpha
        X_train_ = X_train_[:, fs_mask]
        X_test_ = X_test_[:, fs_mask]
    else:
        raise ValueError("Invalid REX parameter `fs` value passed: {0}".format(fs))

    if lagger == 'SEQUENTIAL':
        X_train_, Y_train = prepare_inputs(X_train_, Y_train, window)
        X_test_, Y_test = prepare_inputs(X_test_, Y_test, window)
    elif lagger == 'SIMPLE':
        X_train_, Y_train = simulate_lags(X_train_, Y_train, window)
        X_test_, Y_test = simulate_lags(X_test_, Y_test, window)
    else:
        raise ValueError("Invalid REX parameter `lagger` value passed: {0}".format(diff_type))

    model = model_skeleton(**model_kwg)

    model.fit(X_train=X_train_, Y_train=Y_train.reshape(-1, 1),
              X_val=X_test_, Y_val=Y_test.reshape(-1, 1))

    y_hat_train = model.predict(X=X_train_)
    y_hat_test = model.predict(X=X_test_)

    if diff_type == 'PCT':
        yy_train = Y_train + 1
        yy_train = yy_train.numpy().flatten() * _Y_train[window:-1]
        y_hat_train = y_hat_train + 1
        y_hat_train = y_hat_train.flatten() * _Y_train[window:-1]
        yy_test = Y_test + 1
        yy_test = yy_test.numpy().flatten() * _Y_test[window:-1]
        y_hat_test = y_hat_test + 1
        y_hat_test = y_hat_test.flatten() * _Y_test[window:-1]
    elif diff_type == 'DIFF':
        yy_train = Y_train.numpy().flatten() + _Y_train[window:-1]
        y_hat_train = y_hat_train.flatten() + _Y_train[window:-1]
        yy_test = Y_test.numpy().flatten() + _Y_test[window:-1]
        y_hat_test = y_hat_test.flatten() + _Y_test[window:-1]
    else:
        raise ValueError("Invalid REX parameter `diff_type` value passed: {0}".format(diff_type))

    results_train = check_up(yy_train, y_hat_train, None, X_train_)
    results_test = check_up(yy_test, y_hat_test, None, X_test_)

    results_train['sample'] = 'train'
    results_test['sample'] = 'test'

    results_train = pandas.DataFrame(pandas.Series(results_train))
    results_test = pandas.DataFrame(pandas.Series(results_test))

    results_train = results_train.T
    results_test = results_test.T

    results_train.index = [ex]
    results_test.index = [ex]

    results_train.to_csv('./reported.csv', mode='a', header=False)
    results_test.to_csv('./reported.csv', mode='a', header=False)


def nrex(X_train, Y_train, X_test, Y_test, dmr, pre, fs, model_skeleton, model_kwg, rs):

    if dmr == 'NO':
        X_train_ = X_train
        X_test_ = X_test
    elif dmr == 'PCA_MLE':
        # proj_rate = 0.50  # 0.75   0.5   0.25  'mle'
        # njv = int(X_train_.shape[1] * proj_rate)
        njv = 'mle'
        # njv = 0.75  # 0.75  0.5  0.25
        projector = PCA(n_components=njv, svd_solver='full', random_state=rs)
        projector.fit(X=X_train)
        X_train_ = projector.transform(X_train)
        X_test_ = projector.transform(X_test)
    else:
        raise ValueError("Invalid REX parameter `dmr` value passed: {0}".format(dmr))

    if pre == 'NO':
        pass
    elif pre == 'STD_SCAL':
        scaler = StandardScaler()
        scaler.fit(X=X_train)
        X_train_ = scaler.transform(X_train_)
        X_test_ = scaler.transform(X_test_)
    else:
        raise ValueError("Invalid REX parameter `pre` value passed: {0}".format(pre))

    if fs == 'NO':
        pass
    elif fs == 'KT_05':
        alpha = 0.05
        values = numpy.array([kendalltau(x=X_train_[:, j], y=Y_train)[1] for j in range(X_train_.shape[1])])
        fs_mask = values <= alpha
        X_train_ = X_train_[:, fs_mask]
        X_test_ = X_test_[:, fs_mask]
    else:
        raise ValueError("Invalid REX parameter `fs` value passed: {0}".format(fs))

    model = model_skeleton(**model_kwg)

    model.fit(X=X_train_, y=Y_train.flatten())

    y_hat_train = model.predict(X=X_train_)
    y_hat_test = model.predict(X=X_test_)

    results_train = check_up(Y_train.flatten(), y_hat_train.flatten(), None, X_train_)
    results_test = check_up(Y_test.flatten(), y_hat_test.flatten(), None, X_test_)

    results_train['sample'] = 'train'
    results_test['sample'] = 'test'

    results_train = pandas.DataFrame(pandas.Series(results_train))
    results_test = pandas.DataFrame(pandas.Series(results_test))

    results_train.T.to_csv('./reported.csv', mode='a', header=False)
    results_test.T.to_csv('./reported.csv', mode='a', header=False)
