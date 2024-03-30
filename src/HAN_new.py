#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/25 14:47
# @Author  : Fun.
# @File    : HAN_new.py
# @Software: PyCharm

import torch.nn as nn
from src.Embed_layer import GloveEmbeddingLayer
from src.sent_att_model_new import SentenceAttNet
from src.att_layer import ConversationModel
from src.classifier_layer import ClassifierLayer

class HAN(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, vocab_size, word_embed_dim, char_embed_dim, max_word_length,
                 num_classes, pretrained_word2vec_path):
        super(HAN, self).__init__()
        self.embedding_layer = GloveEmbeddingLayer(pretrained_word2vec_path, vocab_size, word_embed_dim, char_embed_dim, max_word_length)
        self.sentence_representation_layer = SentenceAttNet(hidden_size=sent_hidden_size)
        self.attention_layer = ConversationModel(2 * sent_hidden_size, sent_hidden_size)
        self.classifier_layer = ClassifierLayer(2 * sent_hidden_size, num_classes)

    def forward(self, word_inputs, char_inputs):
        embedded_sentences = self.embedding_layer(word_inputs, char_inputs)
        sentence_representations = self.sentence_representation_layer(embedded_sentences)
        attentive_sentence_representation = self.attention_layer(sentence_representations)
        output = self.classifier_layer(attentive_sentence_representation)
        return output

# Example usage:
# model = HAN(word_hidden_size, sent_hidden_size, vocab_size, word_embed_dim, char_embed_dim, max_word_length,
#             num_classes, pretrained_word2vec_path)
# output = model(word_inputs, char_inputs)