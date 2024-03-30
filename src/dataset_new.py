#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:46
# @Author  : Fun.
# @File    : dataset_new.py
# @Software: PyCharm

from torch.utils.data import Dataset
from src.utils import map_label, build_load_dictionary
import pkuseg

from torch.utils.data import Dataset
from src.utils import map_label, build_load_dictionary, get_max_lengths
import pkuseg

class MyDataset(Dataset):
    def __init__(self, data_path, stop_words_path, max_length_word, max_length_character):
        self.data_path = data_path
        self.max_length_word = max_length_word
        self.max_length_character = max_length_character
        self.segmenter = pkuseg.pkuseg(user_dict='data/user_dic.txt', model_name='medicine')
        self.word_to_idx = {}
        self.char_to_idx = {}
        self.vocab_size = 0
        self.num_classes = 0
        self.data = []
        self.labels = []
        self.stop_words = self.load_stop_words(stop_words_path)
        self.load_data()

    def load_stop_words(self, stop_words_path):
        stop_words = set()
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stop_words.add(word)
        return stop_words

    def load_data(self):
        with open(self.data_path, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                sentence1 = parts[0]
                sentence2 = parts[1]
                label = map_label(parts[3])
                self.data.append((sentence1, sentence2))
                self.labels.append(label)
                self.update_vocab(sentence1)
                self.update_vocab(sentence2)

        self.word_to_idx, self.char_to_idx = build_load_dictionary(self.data)
        self.vocab_size = len(self.word_to_idx)
        self.num_classes = len(set(self.labels))

    def update_vocab(self, sentence):
        words = [word for word in self.segmenter.cut(sentence) if word not in self.stop_words]
        for word in words:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

    def __len__(self):
        return len(self.data)

    def load_stop_words(self, stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().strip().split('\n'))
        return stop_words

    def pad_sequence(self, sequence, max_length, padding_value=0):
        padded_sequence = sequence[:max_length] + [padding_value] * (max_length - len(sequence))
        return padded_sequence

    def __getitem__(self, idx):
        max_length_word, max_length_character = get_max_lengths(self.data_path)
        sentence1, sentence2 = self.data[idx]

        # Remove stop words
        words1 = [word for word in self.segmenter.cut(sentence1) if word not in self.stop_words]
        words2 = [word for word in self.segmenter.cut(sentence2) if word not in self.stop_words]

        print("Original sentences:")
        print("Sentence 1:", sentence1)
        print("Sentence 2:", sentence2)

        print("Words after stop words removal:")
        print("Words 1:", words1)
        print("Words 2:", words2)

        # Convert words to indices
        word_indices1 = [self.word_to_idx.get(word, self.word_to_idx['UNK']) for word in words1]
        word_indices2 = [self.word_to_idx.get(word, self.word_to_idx['UNK']) for word in words2]

        print("Word indices:")
        print("Word indices 1:", word_indices1)
        print("Word indices 2:", word_indices2)

        # Pad word indices to the same length
        word_indices1_padded = self.pad_sequence(word_indices1, self.max_length_word, self.word_to_idx['PAD'])
        word_indices2_padded = self.pad_sequence(word_indices2, self.max_length_word, self.word_to_idx['PAD'])

        print("Padded word indices:")
        print("Padded word indices 1:", word_indices1_padded)
        print("Padded word indices 2:", word_indices2_padded)

        # Character-level encoding
        char_indices1 = [[self.char_to_idx.get(char, self.char_to_idx['UNK']) for char in word] for word in words1]
        char_indices2 = [[self.char_to_idx.get(char, self.char_to_idx['UNK']) for char in word] for word in words2]

        # Flatten the character indices
        char_indices1_flat = [char_idx for sublist in char_indices1 for char_idx in sublist]
        char_indices2_flat = [char_idx for sublist in char_indices2 for char_idx in sublist]

        print("Character indices:")
        print("Character indices 1:", char_indices1)
        print("Character indices 2:", char_indices2)

        print("Flattened character indices:")
        print("Flattened character indices 1:", char_indices1_flat)
        print("Flattened character indices 2:", char_indices2_flat)

        # Pad character indices to the same length
        char_indices1_padded = self.pad_sequence(char_indices1_flat, self.max_length_character, self.char_to_idx['PAD'])
        char_indices2_padded = self.pad_sequence(char_indices2_flat, self.max_length_character, self.char_to_idx['PAD'])

        print("Padded character indices:")
        print("Padded character indices 1:", char_indices1_padded)
        print("Padded character indices 2:", char_indices2_padded)

        return (word_indices1_padded, word_indices2_padded, char_indices1_padded, char_indices2_padded), self.labels[
            idx]
