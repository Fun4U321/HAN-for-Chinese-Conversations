#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:45
# @Author  : Fun.
# @File    : utils.py
# @Software: PyCharm
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn.utils.rnn as rnn_utils
import torch
import csv
import pkuseg
from torch.utils.data import default_collate
csv.field_size_limit(2147483647)
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []

    seg = pkuseg.pkuseg()

    with open(data_path, encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = seg.cut(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = seg.cut(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8 * len(sorted_word_length))], sorted_sent_length[
        int(0.8 * len(sorted_sent_length))]

def map_label(label):
    label_mapping = {"支持": 1, "反对": 2, "补充": 3, "质疑": 4}
    return label_mapping.get(label, 0)

def build_load_dictionary(data):
    """
    从输入的句子中提取出单词/字符字典，并添加 UNK 标记和 PAD 标记。
    Args:
        data (list of tuples): List of tuples where each tuple contains (sentence1, sentence2).

    Returns:
        word_to_idx (dict): Dictionary mapping words to their indices.
        char_to_idx (dict): Dictionary mapping characters to their indices.
    """
    word_to_idx = {'UNK': 0}  # 设置UNK的索引为0
    char_to_idx = {'UNK': 0}  # 设置UNK的索引为0
    word_idx = 1  # 从1开始索引单词
    char_idx = 1  # 从1开始索引字符

    segmenter = pkuseg.pkuseg(model_name='medicine')

    for sentence1, sentence2 in data:
        # Tokenize words in both sentences
        words1 = segmenter.cut(sentence1)
        words2 = segmenter.cut(sentence2)

        # Build word dictionary
        for word in words1:
            if word not in word_to_idx:
                word_to_idx[word] = word_idx
                word_idx += 1
        for word in words2:
            if word not in word_to_idx:
                word_to_idx[word] = word_idx
                word_idx += 1

        # Build character dictionary
        for char in ''.join(words1 + words2):
            if char not in char_to_idx:
                char_to_idx[char] = char_idx
                char_idx += 1

    # Add PAD token for padding
    word_to_idx['PAD'] = len(word_to_idx)
    char_to_idx['PAD'] = len(char_to_idx)

    return word_to_idx, char_to_idx

def my_collate_fn(batch):
    # Make sure each element in the batch contains three elements
    if len(batch[0]) == 2:
        # If not, add None for the missing element
        batch = [(elem[0], None, elem[1]) for elem in batch]
    return default_collate(batch)

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)






