# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:58:03 2018

@author: ladis

Saves prepared dataset into two files as test set and train set
in a format like the following:
This is english sentence.:::1
This is not english sentence.:::0
"""

import glob
import numpy as np
import pandas as pd
import pickle
import csv

#noneng_langs = ["spa", "deu", "fra", "ita"]
noneng_langs = ["deu"]   
    
def create_feature_sets_and_labels(pattern, number_of_files, train_size_percent):
    eng_rows, noneng_rows = get_rows(pattern, number_of_files)
    
    print(eng_rows.groupby(["lang"]).count())
    print(noneng_rows.groupby(["lang"]).count())
    print("--------------------------------")
    numOfEng = len(eng_rows)
    numOfNoneng = len(noneng_rows)
    print("Eng_rows total: ", numOfEng)
    print("Noneng_rows total: ", numOfNoneng)
    print("Total: ", numOfEng + numOfNoneng)
    
    eng_sentences, noneng_sentences = get_only_sentences(eng_rows, noneng_rows)
    np.random.shuffle(noneng_sentences)
    eng_sentences, noneng_sentences = equalize_length_to_lower(eng_sentences, noneng_sentences)
    
    trainsize = int(train_size_percent / 100 * len(eng_sentences))
    return eng_sentences[:trainsize], noneng_sentences[:trainsize], eng_sentences[trainsize:], noneng_sentences[trainsize:]
    
def get_rows(pattern, number_of_files=10):
    files = glob.glob(pattern)
    
    english_rows = pd.DataFrame()
    nonenglish_rows = pd.DataFrame()
    
    for index, file_name in enumerate(files):
        if index == number_of_files:
            break
        
        eng_rows, noneng_rows = extract_rows(file_name)
        
        english_rows = pd.concat([english_rows, eng_rows])
        nonenglish_rows = pd.concat([nonenglish_rows, noneng_rows])
    return english_rows, nonenglish_rows

def equalize_length_to_lower(eng_sentences, noneng_sentences):
    minlen = min(len(eng_sentences), len(noneng_sentences))
    return eng_sentences[:minlen], noneng_sentences[:minlen]

def get_only_sentences(eng_rows, noneng_rows):
    english_sentences = eng_rows.iloc[:,2]
    nonenglish_sentences = noneng_rows.iloc[:,2]
    return np.array(english_sentences), np.array(nonenglish_sentences)
           
def extract_rows(file_name):
    english_sentences = []
    nonenglish_sentences = []
    
    table = pd.read_table(file_name, encoding="ISO-8859-1", delimiter="\t", quoting=csv.QUOTE_NONE, names=["index", "lang", "text"])
    
    english_sentences = table[(table["lang"] == "eng") & (table["text"].notnull())]
    nonenglish_sentences = table[(table["lang"].isin(noneng_langs)) & (table["text"].notnull())]
    
    return english_sentences, nonenglish_sentences
    
# shuffles both arrays and retains their indices in both arrays to be the same
def unison_shuffled_copies(a, b):
    assert len(a) == len(b) 
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == "__main__":
    number_of_files = 30 #437
    train_size_percent = 80
    
    train_eng, train_noneng, test_eng, test_noneng = \
        create_feature_sets_and_labels(r'C:\Users\ladis\OneDrive\Plocha\SentencesDataset\sentences*', \
                                       number_of_files, train_size_percent)
        
    train_eng_y = np.ones((len(train_eng,)), dtype=np.int32)
    train_noneng_y = np.zeros((len(train_noneng,)), dtype=np.int32)
    train_y = np.concatenate((train_eng_y, train_noneng_y))    
    
    test_eng_y = np.ones((len(test_eng,)), dtype=np.int32)
    test_noneng_y = np.zeros((len(test_noneng,)), dtype=np.int32)
    test_y = np.concatenate((test_eng_y, test_noneng_y))    
        
    train = np.concatenate((train_eng, train_noneng))   
    test = np.concatenate((test_eng, test_noneng)) 
    
    train, train_y = unison_shuffled_copies(train, train_y)
    test, test_y = unison_shuffled_copies(test, test_y)
    
    train = np.column_stack((train, train_y))   
    test = np.column_stack((test, test_y)) 
    
    print(len(train_eng))
    print(len(train_noneng))
    print(len(test_eng))
    print(len(test_noneng))
    print(len(train))
    print(len(test))
    print(train.shape)
    
    np.savetxt("train_set_1_80p.csv", train, delimiter=":::", fmt="%s", encoding="ISO-8859-1")
    np.savetxt("test_set_1_20p.csv", test, delimiter=":::", fmt="%s", encoding="ISO-8859-1")
    
    #with open('multilang_sentences_5.pickle', 'wb') as f:
    #    pickle.dump([train_eng, train_noneng, test_eng, test_noneng], f)
    #    print("done")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        