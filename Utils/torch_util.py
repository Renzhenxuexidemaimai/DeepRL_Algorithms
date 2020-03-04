#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午3:32

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor
DOUBLE = torch.DoubleTensor
LONG = torch.LongTensor

def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_params(model: nn.Module):
    """
    get tensor flatted parameters from model
    :param model:
    :return: tensor
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):
    """
    get flatted grad of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])


def set_flat_params(model, flat_params):
    """
    set tensor flatted parameters to model
    :param model:
    :param flat_params: tensor
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
