# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:32:24 2018

@author: ladis
"""

import glob
import numpy as np
import pandas as pd
import pickle
import csv

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
    
    noneng_langs = ["spa", "deu", "fra", "ita"]
    
    english_sentences = table[(table["lang"] == "eng") & (table["text"].notnull())]
    nonenglish_sentences = table[(table["lang"].isin(noneng_langs)) & (table["text"].notnull())]
    
    return english_sentences, nonenglish_sentences
    

if __name__ == "__main__":
    number_of_files = 5 #437
    train_size_percent = 80
    
    train_eng, train_noneng, test_eng, test_noneng = \
        create_feature_sets_and_labels(r'C:\Users\ladis\OneDrive\Plocha\SentencesDataset\sentences*', \
                                       number_of_files, train_size_percent)
    print(len(train_eng))
    print(len(train_noneng))
    print(len(test_eng))
    print(len(test_noneng))
    
    #with open('multilang_sentences_5.pickle', 'wb') as f:
    #    pickle.dump([train_eng, train_noneng, test_eng, test_noneng], f)
    #    print("done")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        