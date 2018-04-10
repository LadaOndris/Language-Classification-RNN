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

from tensorflow.python.ops import rnn, rnn_cell

savePath = "saved/4"

hm_epochs = 5
n_classes = 2
batch_size = 64

rnn_size = 512

x = tf.placeholder("float", [None, 100, 255]) 
y = tf.placeholder("float")

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes]), dtype=tf.float32),
             'biases':tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)}
    
    X_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    lstm_cell = rnn_cell.BasicRNNCell(rnn_size, reuse=None)
    outputs, states = rnn.static_rnn(lstm_cell, X_seqs, dtype=tf.float32)
    
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
    
    return output;

def train_neural_network(x, y, train_x, train_y, test_x, test_y):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    saver = tf.train.Saver(max_to_keep=0)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_batches = int(len(train_x) / batch_size)
        
        for epoch in range(hm_epochs):
            startTime = time.time()
            epoch_loss = 0
            
            batches_x = [train_x[k:k+batch_size] for k in range(n_batches)]
            batches_y = [train_y[k:k+batch_size] for k in range(n_batches)]
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            print('Epoch loss', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
            endTime = time.time()
            print("Training current epoch took {0} minuts".format(get_elapsed_time(startTime, endTime)))
        
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            result = accuracy.eval({x:test_x, y:test_y})
            print('Accuracy:', result)
            with open(savePath+"/epochs-log.txt", "a+") as f:
                f.write("Epoch {0}, accuracy: {1}\n".format(epoch+1, result))
            
            
            saver.save(sess, savePath+"/model", global_step=epoch)
        
        
def prepare_data(eng, noneng):
    eng_x = sentencesToArrayOfASCII(eng)
    noneng_x = sentencesToArrayOfASCII(noneng)
    
    eng_y = [[1, 0] for i in range(len(eng_x))]
    noneng_y = [[0,1] for i in range(len(noneng_x))]
    
    all_x = np.concatenate((eng_x, noneng_x))
    all_y = np.concatenate((eng_y, noneng_y))
    
    all_x, all_y = unison_shuffled_copies(all_x, all_y)
   
    return all_x, all_y
      
# takes array of sentences as input and returns those sentences represented in binary values 
# every letter is reprezented as an array of length 255 (ASCII values) 
# returns an array of shape ((len(sentences_as_strings),100, 255)     
def sentencesToArrayOfASCII(sentences_as_strings):
    sentences = np.ndarray((len(sentences_as_strings), 100, 255), dtype=np.float32)
    for sentence_index, sentence in enumerate(sentences_as_strings):
        sentences[sentence_index] = sentenceToBinary(sentence)
    return sentences
            

# takes a single sentence as input and returns the sentence as an array of shape (100,255)
# for every sentence returns 100 characters long array (either truncates the sentence of pads zeros)
def sentenceToBinary(sentence_as_string):
    sentence_chars = np.zeros((100,255), dtype=np.float32)
    for index, char_as_number in enumerate(map(ord, list(sentence_as_string))):
        if index >= 100:
            break
        sentence_chars[index][char_as_number] = 1
    return np.array(sentence_chars)
          

# shuffles both arrays
def unison_shuffled_copies(a, b):
    assert len(a) == len(b) 
    p = np.random.permutation(len(a))
    return a[p], b[p]

# gets elapsed time in minutes from start to end
def get_elapsed_time(start, end):
    return round((end - float(start)) / 60, 2)

startTime = time.time()  
filename = 'multilang_sentences_5.pickle'
if os.path.getsize(filename) > 0:      
    if not os.path.exists(savePath):
        os.makedirs(savePath)     
    with open(filename, 'rb') as f:
        train_eng, train_noneng, test_eng, test_noneng = pickle.load(f)
        train_x, train_y = prepare_data(train_eng, train_noneng)
        test_x, test_y = prepare_data(test_eng, test_noneng)
        
        
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    print("Training started.")
    endTime = time.time()
    print("Elapsed time since execution: ", get_elapsed_time(startTime, endTime))
    train_neural_network(x, y, train_x, train_y, test_x, test_y)
else:
    print("File is empty")
    























