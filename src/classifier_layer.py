#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 20:31
# @Author  : Fun.
# @File    : classifier_layer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierLayer(nn.Module):
    def __init__(self, input_size, num_topics):
        super(ClassifierLayer, self).__init__()
        self.linear = nn.Linear(input_size, num_topics)

    def forward(self, input_representation):
        # 应用线性变换
        linear_output = self.linear(input_representation)
        # 应用softmax计算条件概率
        probabilities = F.softmax(linear_output, dim=1)
        return probabilities

# 示例用法：
# 假设input_representation是从具有注意力机制的BLSTM得到的最终表示S_alpha
# input_representation.shape应该是(batch_size, representation_size)
# num_topics是分类的主题数量
# classifier = ClassifierLayer(representation_size, num_topics)
# probabilities = classifier(input_representation)
# probabilities.shape将是(batch_size, num_topics)
