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

savePath = "saved/13"
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
hm_epochs = 5 
n_classes = 2
batch_size = 64
total_batches = int(28992 / batch_size) # 453
rnn_size = 512
n_accuracy_testing_per_epoch = 3
x = tf.placeholder("float", [None, 100, 255]) 
y = tf.placeholder("float")
learning_rate = tf.placeholder(tf.float32, shape=[])

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes]), dtype=tf.float32),
             'biases':tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)}
    
    X_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    lstm_cell = rnn_cell.BasicRNNCell(rnn_size, reuse=None)
    outputs, states = rnn.static_rnn(lstm_cell, X_seqs, dtype=tf.float32)
    
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
    
    return output;

def train_neural_network(x, y, learningRate, test_x, test_y, train_x, train_y):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    saver = tf.train.Saver(max_to_keep=0)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess,  tf.train.latest_checkpoint("saved/10/"))
        
        errors = []
        accuracyTrain = []
        accuracyTest = []
        
        for epoch in range(hm_epochs):
            print("Epoch started")
            epoch_loss = 0
            #epochStartTime = time.time()
            
            with open('train_set_1_80p.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = np.ndarray((batch_size, 100, 255), dtype=np.float32)
                batch_y = np.ndarray((batch_size, 2), dtype=np.float32)
                batch_x_length = 0
                batch_y_length = 0
                batches_run = 0
                
                for line in f:
                    if batches_run >= total_batches:
                        break
                    
                    x_string, y_string = parse_line(line)
                    single_input_x, single_input_y = prepare_input(x_string, y_string)
                    
                    batch_x[batch_x_length] = single_input_x
                    batch_y[batch_y_length] = single_input_y
                    batch_x_length += 1
                    batch_y_length += 1
                    
                    if batch_x_length >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, learning_rate: learningRate})
                        epoch_loss += c
                        errors.append(c)
                        #####
                        
                        if batches_run % (int(total_batches / n_accuracy_testing_per_epoch)) == 0:
                            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
                            # testing set accuracy
                            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                            testAccuracy = accuracy.eval({x:test_x, y:test_y})
                            print('Test accuracy:', testAccuracy)
                            
                            #training set accuracy
                            trainAccuracy = accuracy.eval({x:train_x, y:train_y})
                            print('Train accuracy:', trainAccuracy)
                            
                            accuracyTest.append(testAccuracy)
                            accuracyTrain.append(trainAccuracy)
                        ####
                        
                        batch_x = np.ndarray((batch_size, 100, 255), dtype=np.float32)
                        batch_y = np.ndarray((batch_size, 2), dtype=np.float32)
                        batch_x_length = 0
                        batch_y_length = 0
                        batches_run += 1
                        print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch+1,'| Batch Loss:',c,)
                        
            
# =============================================================================
#             errors.append(epoch_loss)
#             print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
#             
#             epochEndTime = time.time()
#             print("Training current epoch took {0} minuts".format(get_elapsed_time(epochStartTime, epochEndTime)))
#             
#             # testing set accuracy
#             correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#             accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#             testAccuracy = accuracy.eval({x:test_x, y:test_y})
#             print('Test accuracy:', testAccuracy)
#             
#             #training set accuracy
#             trainAccuracy = accuracy.eval({x:train_x, y:train_y})
#             print('Train accuracy:', trainAccuracy)
#             
#             accuracyTest.append(testAccuracy)
#             accuracyTrain.append(trainAccuracy)
# =============================================================================
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            result = accuracy.eval({x:test_x, y:test_y})
            print('Epoch test accuracy:', result)
            
            with open(savePath+"/epochs-log.txt", "a+") as f:
                f.write("Epoch {0}, accuracy: {1}, epoch loss: {2}\n".format(epoch+1, result, epoch_loss))
            saver.save(sess, savePath+"/model", global_step=epoch)
            
        plot(errors, accuracyTrain, accuracyTest)
            

def plot(epoch_losses, accuracyTrain, accuracyTest):
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(211)
    plt.plot(epoch_losses)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(accuracyTrain, "r", label="Train accuracy")
    plt.plot(accuracyTest, "b", label="Test accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.show()

def parse_line(line):
    sentence = line.split(':::')[0]
    label = line.split(':::')[1]     
    return sentence, label

def prepare_input(x_string, y_string):
    single_input_x = sentenceToArrayOfASCII(x_string)
    single_input_y = label_to_vector(y_string)
    return single_input_x, single_input_y

def label_to_vector(label):
    if label.strip() == "1":
        return np.array([1, 0])
    else:
        return np.array([0, 1])
        
       
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
    
# gets elapsed time in minutes from start to end
def get_elapsed_time(start, end):
    return round((end - float(start)) / 60, 2)
   
def load_set(path, max_num_of_samples):
    sample_x = np.ndarray((max_num_of_samples, 100, 255), dtype=np.float32)
    sample_y = np.ndarray((max_num_of_samples, 2), dtype=np.float32)
    sample_x_length = 0
    sample_y_length = 0
    
    with open(path, buffering=20000, encoding='latin-1') as f:
        for line in f:
            if sample_x_length >= max_num_of_samples:
                break
            x_string, y_string = parse_line(line)
            single_input_x, single_input_y = prepare_input(x_string, y_string)
            
            sample_x[sample_x_length] = single_input_x
            sample_y[sample_y_length] = single_input_y
            sample_x_length += 1
            sample_y_length += 1
    
    return sample_x, sample_y


test_x, test_y = load_set('test_set_1_20p.csv', 1000)
train_x, train_y = load_set('train_set_1_80p.csv', 1000)
        
train_neural_network(x, y, 0.00005, test_x, test_y, train_x, train_y)
 






















