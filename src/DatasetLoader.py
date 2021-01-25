# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:03:30 2018

@author: Ladislav Ondris


Loads prepared dataset from text files.
On each line of the text file is a single sentence in the following form:

This is an example sentence.:::0

where sentence comes first, then comes the label after the ::: delimeter.
"""

import numpy as np


class DatasetLoader:
    def load_set(self, path, max_num_of_samples, n_classes):
        sample_x = np.ndarray((max_num_of_samples, 100, 255), dtype=np.float32)
        sample_y = np.ndarray((max_num_of_samples, n_classes), dtype=np.float32)
        sample_x_length = 0
        sample_y_length = 0

        with open(path, buffering=20000, encoding='latin-1') as f:
            for line in f:
                if sample_x_length >= max_num_of_samples:
                    break
                single_input_x, single_input_y = self.prepare_input(line)

                sample_x[sample_x_length] = single_input_x
                sample_y[sample_y_length] = single_input_y
                sample_x_length += 1
                sample_y_length += 1

        return sample_x, sample_y

    def parse_line(self, line):
        sentence = line.split(':::')[0]
        label = line.split(':::')[1]
        return sentence, label

    def prepare_input(self, line):
        sentence, label = self.parse_line(line)
        single_input_x = self.sentenceToArrayOfASCII(sentence)
        single_input_y = self.__label_to_vector(label)
        return single_input_x, single_input_y

    # takes a single sentence as input and returns the sentence as an array of shape (100,255)
    # for every sentence returns 100 characters long array (either truncates the sentence or pads zeros)
    def sentenceToArrayOfASCII(self, sentence_as_string):
        sentence_chars = np.zeros((100, 255), dtype=np.float32)
        for index, char_as_number in enumerate(map(ord, list(sentence_as_string))):
            if index >= 100:
                break
            if char_as_number >= 255:
                continue
            sentence_chars[index][char_as_number] = 1
        return np.array(sentence_chars)

    def __label_to_vector(self, label):
        label = label.strip()
        if label == "0":
            return np.array([1, 0, 0, 0, 0])
        elif label == "1":
            return np.array([0, 1, 0, 0, 0])
        elif label == "2":
            return np.array([0, 0, 1, 0, 0])
        elif label == "3":
            return np.array([0, 0, 0, 1, 0])
        else:
            if label != "4":
                print("THERE IS AN UNKNOWN LABEL INDEX", label)
            return np.array([0, 0, 0, 0, 1])
