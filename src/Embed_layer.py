#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 22:11
# @Author  : Fun.
# @File    : Embed_layer.py
# @Software: PyCharm

import torch
import torch.nn as nn

class GloveEmbeddingLayer(nn.Module):
    def __init__(self, glove_path, vocab_size, word_embed_dim, char_embed_dim, max_word_length):
        super(GloveEmbeddingLayer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.char_embedding = nn.Embedding(256, char_embed_dim)  # 256 是 ASCII 字符的数量
        self.char_cnn = nn.Conv1d(char_embed_dim, char_embed_dim, kernel_size=3, padding=1)
        self.char_pool = nn.MaxPool1d(kernel_size=max_word_length)
        self.load_glove_weights(glove_path)

    def load_glove_weights(self, glove_path):
        """
        Load pre-trained GloVe word embeddings.
        Args:
            glove_path (str): Path to the GloVe word embeddings file.
        """
        # 加载 GloVe 词嵌入权重的逻辑

    def forward(self, word_inputs, char_inputs):
        """
        Args:
            word_inputs (LongTensor): LongTensor of shape (batch_size, max_sentence_length)
            char_inputs (LongTensor): LongTensor of shape (batch_size, max_sentence_length, max_word_length)
        Returns:
            sentence_representation (FloatTensor): FloatTensor of shape (batch_size, max_sentence_length, word_embed_dim + char_embed_dim)
        """
        word_embedded_output = self.word_embedding(word_inputs)
        batch_size, max_sentence_length, max_word_length = char_inputs.size()
        char_inputs = char_inputs.view(-1, max_word_length)
        char_embedded_output = self.char_embedding(char_inputs)
        char_embedded_output = char_embedded_output.view(batch_size * max_sentence_length, -1, max_word_length)
        char_cnn_output = self.char_cnn(char_embedded_output)
        char_pool_output = self.char_pool(char_cnn_output)
        char_embedded_output = char_pool_output.squeeze(dim=2)
        char_embedded_output = char_embedded_output.view(batch_size, max_sentence_length, -1)
        sentence_representation = torch.cat((word_embedded_output, char_embedded_output), dim=2)
        return sentence_representation
