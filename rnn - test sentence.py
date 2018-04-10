# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:22:31 2018

@author: ladis
"""



import tensorflow as tf
import pickle
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from tensorflow.python.ops import rnn, rnn_cell

tf.reset_default_graph()

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes]), dtype=tf.float32),
             'biases':tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)}
    
    X_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    lstm_cell = rnn_cell.BasicRNNCell(rnn_size, reuse=None)
    outputs, states = rnn.static_rnn(lstm_cell, X_seqs, dtype=tf.float32)
    
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
    
    return output;

def get_output_for_sentence(x, sentence):
    saver = tf.train.Saver(max_to_keep=0)
    
    with tf.Session() as sess:
        saver.restore(sess,  tf.train.latest_checkpoint("saved/13/"))
        output = tf.argmax(prediction, 1)
        result = output.eval({ x: sentence })
        return result
         
# takes a single sentence as input and returns the sentence as an array of shape (100,255)
# for every sentence returns 100 characters long array (either truncates the sentence of pads zeros)
def sentenceToArrayOfASCII(sentence_as_string):
    sentence_chars = np.zeros((100,255), dtype=np.float32)
    for index, char_as_number in enumerate(map(ord, list(sentence_as_string))):
        if index >= 100:
            break
        if char_as_number >= 255:
            continue
        sentence_chars[index][char_as_number] = 1
    return np.array(sentence_chars)

def prepare_sentence(sentence):
    resultArray = np.ndarray((1, 100, 255), dtype=np.float32)
    sentenceASCII = sentenceToArrayOfASCII(sentence)
    resultArray[0] = sentenceASCII
    return resultArray


n_classes = 2
rnn_size = 512
x = tf.placeholder("float", [None, 100, 255]) 
prediction = recurrent_neural_network(x)   


while True:
    sentence = input("Zadej anglickou nebo německou větu: ")
    sentence = prepare_sentence(sentence)
    result = get_output_for_sentence(x, sentence)
    
    if result == 0:
        print("Anglická věta")
    elif result == 1:
        print("Německá věta")
    else:
        print("Neznámý výsledek")
        
        

 






















