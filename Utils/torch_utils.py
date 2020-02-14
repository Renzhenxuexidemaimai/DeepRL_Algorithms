#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午3:32

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor


def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_parameters(model: nn.Module):
    """
    get flatted parameters from model
    :param model:
    :return:
    """
    return torch.cat([grad.view(-1) for grad in model.paramters()])


def set_flat_params(model, flat_params):
    """
    set flatted parameters to model
    :param model:
    :param flat_params:
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
