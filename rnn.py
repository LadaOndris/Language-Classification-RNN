# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:22:31 2018

@author: Ladislav Ondris


RNN for language classification from sentences.

Given a sentence, the RNN classifies the language of the sentence.
"""

import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell

import DatasetLoader


class Network:
    
    def __init__(self, n_samples, n_epochs, batch_size, n_classes, rnn_size, n_accuracy_testing_per_epoch,
                 saveModelAfterEachEpoch = True, savePath = "/saved", measureEpochTime=True):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.rnn_size = rnn_size
        self.n_accuracy_testing_per_epoch = n_accuracy_testing_per_epoch
        self.saveModelAfterEachEpoch = saveModelAfterEachEpoch
        self.savePath = savePath
        self.measureEpochTime = measureEpochTime
        
        self.total_batches = int(n_samples / batch_size) 
        
        if saveModelAfterEachEpoch:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
                
        self.datasetLoader = DatasetLoader.DatasetLoader()
            

    def recurrent_neural_network(self, x):
        layer = {'weights':tf.Variable(tf.random_normal([self.rnn_size, self.n_classes]), dtype=tf.float32),
                 'biases':tf.Variable(tf.random_normal([self.n_classes]), dtype=tf.float32)}
        
        X_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        lstm_cell = rnn_cell.BasicRNNCell(self.rnn_size, reuse=None)
        outputs, states = rnn.static_rnn(lstm_cell, X_seqs, dtype=tf.float32)
        
        output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
        
        return output;
    
    def train_neural_network(self, x, y, learningRate, test_x, test_y, train_x, train_y, train_set_file_path):
        prediction = self.recurrent_neural_network(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        saver = tf.train.Saver(max_to_keep=0)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            #saver.restore(sess,  tf.train.latest_checkpoint("saved/v3/1"))
            
            errors = []
            accuracyTrain = []
            accuracyTest = []
            
            for epoch in range(self.n_epochs):
                epoch_loss = 0
                if self.measureEpochTime:
                    epochStartTime = time.time()
                
                with open(train_set_file_path, buffering=20000, encoding='latin-1') as f:
                    batch_x = np.ndarray((self.batch_size, 100, 255), dtype=np.float32)
                    batch_y = np.ndarray((self.batch_size, self.n_classes), dtype=np.float32)
                    batch_x_length = 0
                    batch_y_length = 0
                    batches_run = 0
                    
                    for line in f:
                        if batches_run >= self.total_batches:
                            break
                        
                        single_input_x, single_input_y = self.datasetLoader.prepare_input(line)
                        
                        batch_x[batch_x_length] = single_input_x
                        batch_y[batch_y_length] = single_input_y
                        batch_x_length += 1
                        batch_y_length += 1
                        
                        if batch_x_length >= self.batch_size:
                            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, learning_rate: learningRate})
                            epoch_loss += c
                            errors.append(c)
                            
                            #####
                            if batches_run % (int(self.total_batches / self.n_accuracy_testing_per_epoch)) == 0:
                                print('Epoch', epoch + 1, 'completed out of', self.n_epochs, 'loss:', epoch_loss)
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
                            
                            batch_x = np.ndarray((self.batch_size, 100, 255), dtype=np.float32)
                            batch_y = np.ndarray((self.batch_size, self.n_classes), dtype=np.float32)
                            batch_x_length = 0
                            batch_y_length = 0
                            batches_run += 1
                            print('Batch run:',batches_run,'/',self.total_batches,'| Epoch:',epoch+1,'| Batch Loss:',c,)
                            
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                result = accuracy.eval({x:test_x, y:test_y})
                print('Epoch test accuracy:', result)
                
                with open(self.savePath+"/epochs-log.txt", "a+") as f:
                    f.write("Epoch {0}, accuracy: {1}, epoch loss: {2}\n".format(epoch+1, result, epoch_loss))
                saver.save(sess, self.savePath+"/model", global_step=epoch)
                
                if self.measureEpochTime:
                     epochEndTime = time.time()
                     print("Epoch was traing",self.get_elapsed_time(epochStartTime, epochEndTime),"minuts.")
                
            self.plot_result_of_training(errors, accuracyTrain, accuracyTest)
    
    # gets elapsed time in minutes from start to end
    def get_elapsed_time(self, start, end):
        return round((end - float(start)) / 60, 2)
            
    def plot_result_of_training(self, epoch_losses, accuracyTrain, accuracyTest):
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



tf.reset_default_graph()

#total_batches = int(28992 / batch_size) # 453

x = tf.placeholder("float", [None, 100, 255]) 
y = tf.placeholder("float")
learning_rate = tf.placeholder(tf.float32, shape=[])

datasetLoader = DatasetLoader.DatasetLoader()
test_x, test_y = datasetLoader.load_set('test_set_all_langs_30_files.csv', 1000, 5)
train_x, train_y = datasetLoader.load_set('train_set_all_langs_30_files.csv', 1000, 5)
#23085      30 files
#263615     100 files
network = Network(n_samples=9984, n_epochs=5, batch_size=64, n_classes=5, rnn_size=512,\
                  n_accuracy_testing_per_epoch=5, savePath="saved/v3/2")
network.train_neural_network(x, y, 0.00005, test_x, test_y, train_x, train_y, 'train_set_all_langs_30_files.csv')


            

 






















