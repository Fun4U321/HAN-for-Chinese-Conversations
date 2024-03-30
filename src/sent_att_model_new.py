#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 21:44
# @Author  : Fun.
# @File    : sent_att_model_new.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import matrix_mul, element_wise_mul


class SentenceAttNet(nn.Module):
    def __init__(self, hidden_size=50):
        super(SentenceAttNet, self).__init__()
        self.sentence_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.sentence_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sentence_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = input
        output = matrix_mul(f_output, self.sentence_weight, self.sentence_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0))
        return output, h_output
