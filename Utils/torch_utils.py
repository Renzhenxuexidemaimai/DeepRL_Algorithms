#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午3:32

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor

def to_device(*args):
    return [arg.to(device) for arg in args]
