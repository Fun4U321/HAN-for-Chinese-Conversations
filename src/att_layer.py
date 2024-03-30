#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 20:27
# @Author  : Fun.
# @File    : att_layer.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceRepresentationLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentenceRepresentationLayer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, input):
        output, _ = self.gru(input)
        return output

class BLSTMAAttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BLSTMAAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, input):
        output, _ = self.blstm(input)
        forward_output = output[:, :, :self.hidden_size]  # 使用 self.hidden_size
        backward_output = output[:, :, self.hidden_size:]  # 使用 self.hidden_size
        attention_weights = F.softmax(self.linear(output), dim=1)
        attentive_output = torch.sum(attention_weights * output, dim=1)
        return attentive_output


class ConversationModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConversationModel, self).__init__()
        self.sentence_representation = SentenceRepresentationLayer(input_size, hidden_size)
        self.attention_layer = BLSTMAAttentionLayer(2 * hidden_size, hidden_size)

    def forward(self, input):
        sentence_representations = self.sentence_representation(input)
        conversation_representation = self.attention_layer(sentence_representations)
        return conversation_representation

# Usage example:
model = ConversationModel(input_size=100, hidden_size=50)
input_data = torch.randn(5, 10, 100)  # Example input: batch size 5, sequence length 10, input size 100
output = model(input_data)
print(output.shape)  # Expected output shape: (5, 100)
