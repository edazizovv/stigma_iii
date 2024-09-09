#


#
import torch
from torch import nn


#


#
nn_essence = {None: None,
              'None': None,
              'Linear': nn.Linear,
              'LSTM': nn.LSTM,
              'BatchNorm1d': nn.BatchNorm1d,
              'LeakyReLU': nn.LeakyReLU,
              'ReLU': nn.ReLU,
              'Mish': nn.Mish,
              'Swish': nn.SiLU,
              'Gelu': nn.GELU,
              'Silu': nn.SiLU,
              'Adamax': torch.optim.Adamax,
              'MSELoss': nn.MSELoss}


def translate_nn_kwargs(str_kwargs):

    valid_kwargs = dict(str_kwargs)
    valid_kwargs['layers'] = [nn_essence[x] for x in valid_kwargs['layers']]
    valid_kwargs['batchnorms'] = [nn_essence[x] for x in valid_kwargs['batchnorms']]
    valid_kwargs['activators'] = [nn_essence[x] for x in valid_kwargs['activators']]
    valid_kwargs['optimiser'] = nn_essence[valid_kwargs['optimiser']]
    valid_kwargs['loss_function'] = nn_essence[valid_kwargs['loss_function']]

    return valid_kwargs
