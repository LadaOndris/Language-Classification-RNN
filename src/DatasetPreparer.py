# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:58:03 2018

@author: Ladislav Ondris

Saves prepared dataset into two files as test set and train set
in a format like the following:
This is english sentence.:::1
This is not english sentence.:::0
"""

import glob
import numpy as np
import pandas as pd
import csv


class DatasetPreparer:
    def __init__(self, lang_groups):
        self.lang_groups = lang_groups

    def get_train_and_test_data(self, target_folder, number_of_files, train_size_percentage):
        train_groups, test_groups = \
            self.create_feature_sets_and_labels(r'{0}*'.format(target_folder),
                                                number_of_files, train_size_percentage)

        train_groups_y = [np.full((len(group, )), index, dtype=np.int32) for index, group in enumerate(train_groups)]
        train_y = np.array(train_groups_y).reshape((-1,))

        test_groups_y = [np.full((len(group, )), index, dtype=np.int32) for index, group in enumerate(test_groups)]
        test_y = np.array(test_groups_y).reshape((-1,))

        train = np.array(train_groups).reshape((-1,))
        test = np.array(test_groups).reshape((-1,))

        train, train_y = self.unison_shuffled_copies(train, train_y)
        test, test_y = self.unison_shuffled_copies(test, test_y)

        train = np.column_stack((train, train_y))
        test = np.column_stack((test, test_y))

        return train, test

    def create_feature_sets_and_labels(self, pattern, number_of_files, train_size_percentage):
        grouped_rows = self.get_rows(pattern, number_of_files)

        #        print(eng_rows.groupby(["lang"]).count())
        #        print(noneng_rows.groupby(["lang"]).count())
        #        print("--------------------------------")
        #        numOfEng = len(eng_rows)
        #        numOfNoneng = len(noneng_rows)
        #        print("Eng_rows total: ", numOfEng)
        #        print("Noneng_rows total: ", numOfNoneng)
        #        print("Total: ", numOfEng + numOfNoneng)

        grouped_sentences = self.get_only_sentences(grouped_rows)
        [np.random.shuffle(group) for group in grouped_sentences]
        grouped_sentences = self.equalize_length_to_lower(grouped_sentences)

        trainsize = int(train_size_percentage / 100 * len(grouped_sentences[0]))
        return [group[:trainsize] for group in grouped_sentences], [group[trainsize:] for group in grouped_sentences],

    def get_rows(self, pattern, number_of_files=10):
        files = glob.glob(pattern)

        frames = [pd.DataFrame() for n in range(len(self.lang_groups))]

        for index, file_name in enumerate(files):
            if index == number_of_files:
                break

            grouped_rows = self.grouped_rows_from_file(file_name)

            for i, (frame, rows) in enumerate(zip(frames, grouped_rows)):
                frames[i] = pd.concat([frame, rows])

        return frames

    def equalize_length_to_lower(self, arrays):
        minlen = min(map(len, arrays))
        return [array[:minlen] for array in arrays]

    def get_only_sentences(self, grouped_rows):
        return [np.array(group.iloc[:, 2]) for group in grouped_rows]

    def grouped_rows_from_file(self, file_name):
        table = pd.read_table(file_name, encoding="ISO-8859-1", delimiter="\t", quoting=csv.QUOTE_NONE,
                              names=["index", "lang", "text"])
        rows = [table[(table["lang"].isin(lang_group)) & (table["text"].notnull())] for lang_group in self.lang_groups]
        return rows

    # shuffles both arrays and retains their indices in both arrays to be the same
    def unison_shuffled_copies(self, array1, array2):
        assert len(array1) == len(array2)
        p = np.random.permutation(len(array1))
        return array1[p], array2[p]


if __name__ == "__main__":
    target_folder = "??????"
    langs = [["eng"], ["deu"], ["fra"], ["ita"], ["spa"]]
    number_of_files = 100
    train_set_percentage = 95
    datasetPreparer = DatasetPreparer(langs)
    train_set, test_set = datasetPreparer.get_train_and_test_data(target_folder, number_of_files, train_set_percentage)

    print(train_set.shape)
    print(test_set.shape)
    print(len(train_set))
    print(len(test_set))

    np.savetxt("train_set_all_langs_100_files.csv", train_set, delimiter=":::", fmt="%s", encoding="ISO-8859-1")
    np.savetxt("test_set_all_langs_100_files.csv", test_set, delimiter=":::", fmt="%s", encoding="ISO-8859-1")
