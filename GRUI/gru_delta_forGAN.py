#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:52:13 2018
gru for imputed data
@author: lyh
"""


from __future__ import print_function  # Ensures compatibility of print function between Python 2 and 3
import os  # Operating system-related functions for handling directories
import numpy as np  # Numerical computations library
from sklearn import metrics  # Import metrics module from scikit-learn for model evaluation
import time  # Module to measure execution time
import mygru_cell  # Import custom GRU cell module
import tensorflow as tf  # Import TensorFlow for machine learning tasks
from tensorflow.python.ops import math_ops  # Import math operations from TensorFlow library
tf.set_random_seed(1)   # Set random seed for reproducibility of TensorFlow computations

# Define GRU model class
class grui(object):
    model_name = "GRU_I"  # Set the name of the model

    # Initialization function for the model
    def __init__(self, sess, args, dataset, test_set):
        self.lr = args.lr  # Learning rate from the input arguments
        self.sess = sess  # TensorFlow session
        self.isbatch_normal = args.isBatch_normal  # Batch normalization flag
        self.isNormal = args.isNormal  # Normalization flag
        self.isSlicing = args.isSlicing  # Data slicing flag
        self.dataset = dataset  # Training dataset
        self.test_set = test_set  # Test dataset
        self.epoch = args.epoch  # Number of epochs
        self.batch_size = args.batch_size  # Batch size
        self.n_inputs = args.n_inputs  # Number of inputs
        self.n_steps = dataset.maxLength  # Maximum time steps based on dataset
        self.n_hidden_units = args.n_hidden_units  # Hidden units in GRU layer
        self.n_classes = args.n_classes  # Number of output classes
        self.run_type = args.run_type  # Run type (test/train)
        self.result_path = args.result_path  # Path to store results
        self.model_path = args.model_path  # Path to load a model
        self.log_dir = args.log_dir  # Path to store logs
        self.checkpoint_dir = args.checkpoint_dir  # Path to store checkpoints
        self.num_batches = len(dataset.x) // self.batch_size  # Number of batches in the dataset
        
        # Placeholder definitions for input data, labels, and other variables
        self.keep_prob = tf.placeholder(tf.float32)  # Placeholder for dropout keep probability
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])  # Input data placeholder
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])  # Output labels placeholder
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])  # Mask placeholder for missing values
        self.delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])  # Time gap placeholder
        self.mean = tf.placeholder(tf.float32, [self.n_inputs, ])  # Mean of observations
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])  # Placeholder for last observed values
        self.x_lengths = tf.placeholder(tf.int32, shape=[self.batch_size, ])  # Lengths of each time series sequence

    # Function to define the RNN model
    def RNN(self, X, M, Delta, Mean, Lastvalues, X_lengths, Keep_prob, reuse=False):
        with tf.variable_scope("grui", reuse=reuse):  # Define a variable scope for the GRU
            # Define weight and bias variables for the hidden layer and output layer
            wr_h = tf.get_variable('wr_h', shape=[self.n_inputs, self.n_hidden_units], initializer=tf.random_normal_initializer())
            w_out = tf.get_variable('w_out', shape=[self.n_hidden_units, self.n_classes], initializer=tf.random_normal_initializer())
            br_h = tf.get_variable('br_h', shape=[self.n_hidden_units, ], initializer=tf.constant_initializer(0.001))
            b_out = tf.get_variable('b_out', shape=[self.n_classes, ], initializer=tf.constant_initializer(0.001))
        
            # Reshape last observed values, input data, and time gaps
            Lastvalues = tf.reshape(Lastvalues, [-1, self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            Delta = tf.reshape(Delta, [-1, self.n_inputs])
            
            # Compute rth, which models the decay based on Delta, using the hidden weights and biases
            rth = tf.matmul(Delta, wr_h) + br_h
            rth = math_ops.exp(-tf.maximum(0.0, rth))  # Apply exponential decay to rth
            
            # Concatenate input data X and rth
            X = tf.concat([X, rth], 1)
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs + self.n_hidden_units])  # Reshape concatenated data for RNN input
            
            # Select GRU cell version based on TensorFlow version and initialize RNN
            if "1.5" in tf.__version__ or "1.7" in tf.__version__ :   
                grud_cell = mygru_cell.MyGRUCell15(self.n_hidden_units)
            elif "1.4" in tf.__version__:
                grud_cell = mygru_cell.MyGRUCell4(self.n_hidden_units)
            elif "1.2" in tf.__version__:
                grud_cell = mygru_cell.MyGRUCell2(self.n_hidden_units)

            # Initialize GRU cell state with zeros
            init_state = grud_cell.zero_state(self.batch_size, dtype=tf.float32)
            
            # Run the dynamic RNN with the initialized GRU cell
            outputs, final_state = tf.nn.dynamic_rnn(grud_cell, X_in, initial_state=init_state, sequence_length=X_lengths, time_major=False)
         
            # Output layer transformations
            factor = tf.matrix_diag([1.0/9, 1])  # Define transformation matrix for the output
            tempout = tf.matmul(tf.nn.dropout(final_state, Keep_prob), w_out) + b_out  # Apply dropout and compute output
            results = tf.nn.softmax(tf.matmul(tempout, factor))  # Apply softmax to output results
            return results  # Return the final results

    # Function to build the computational graph for the model
    def build(self):
        self.pred = self.RNN(self.x, self.m, self.delta, self.mean, self.lastvalues, self.x_lengths, self.keep_prob)  # Call RNN function
        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.pred))  # Define cross-entropy loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)  # Define the training operation
        
        # Accuracy computation
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))  # Compare predicted and true labels
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate average accuracy
        self.saver = tf.train.Saver(max_to_keep=None)  # Saver for saving and restoring model checkpoints
        
        # Logging the loss and accuracy
        loss_sum = tf.summary.scalar("loss", self.cross_entropy)
        acc_sum = tf.summary.scalar("acc", self.accuracy)
        self.sum = tf.summary.merge([loss_sum, acc_sum])  # Merge summaries

    # Function to create the model directory
    def model_dir(self, epoch):
        return "{}_{}_{}_{}_{}_{}/epoch{}".format(self.model_name, self.lr, self.batch_size, self.isNormal, self.isbatch_normal, self.isSlicing, epoch)  # Format model directory path
        
    # Save the model at a specific checkpoint
    def save(self, checkpoint_dir, step, epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir(epoch), self.model_name)  # Build the checkpoint path
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)  # Create directory if it doesn't exist
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)  # Save model at the current step

    # Load a saved model from checkpoint
    def load(self, checkpoint_dir, epoch):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir(epoch), self.model_name)  # Build checkpoint path
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # Get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # Get the checkpoint filename
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))  # Restore model from checkpoint
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))  # Extract the checkpoint step number
            print(" [*] Success to read {}".format(ckpt_name))  # Print success message
            return True, counter
        else:
            return False, 0  # Return failure to load model

    # Train the model
    def train(self):
        max_auc = 0.5  # Initialize maximum AUC score
        model_dir2 = "{}_{}_{}_{}_{}_{}".format(self.model_name, self.lr, self.batch_size, self.isNormal, self.isbatch_normal, self.isSlicing)  # Build model directory string
        if not os.path.exists(os.path.join(self.checkpoint_dir, model_dir2)):
            os.makedirs(os.path.join(self.checkpoint_dir, model_dir2))  # Create directory if it doesn't exist
        result_file = open(os.path.join(self.checkpoint_dir, model_dir2, "result"), "a+")  # Open result file for appending

        # Load existing model if available
        if os.path.exists(os.path.join(self.checkpoint_dir, self.model_dir(self.epoch), self.model_name)):
            for nowepoch in range(1, self.epoch + 1):
                print(" [*] Load SUCCESS")
                print("epoch: " + str(nowepoch))
                self.load(self.checkpoint_dir, nowepoch)  # Load model for each epoch
                acc, auc, model_name = self.test(self.test_set, nowepoch)  # Test model on test set
                if auc > max_auc:
                    max_auc = auc  # Update maximum AUC if the new AUC is higher
                result_file.write("epoch: " + str(nowepoch) + "," + str(acc) + "," + str(auc) + "\r\n")  # Write results to file
                print("")
            result_file.close()
            return max_auc  # Return maximum AUC
        else:
            tf.global_variables_initializer().run()  # Initialize TensorFlow variables if model load fails
            counter = 1
            print(" [!] Load failed...")

        start_time = time.time()  # Start the timer
        idx = 0
        epochcount = 0
        dataset = self.dataset
        while epochcount < self.epoch:
            dataset.shuffle(self.batch_size, True)  # Shuffle the dataset for training
            for data_x, data_y, data_mean, data_m, data_delta, data_x_lengths, data_lastvalues, _, _, _ in dataset.nextBatch():
                _, loss, summary_str, acc = self.sess.run([self.train_op, self.cross_entropy, self.sum, self.accuracy], feed_dict={
                    self.x: data_x,
                    self.y: data_y,
                    self.m: data_m,
                    self.delta: data_delta,
                    self.mean: data_mean,
                    self.x_lengths: data_x_lengths,
                    self.lastvalues: data_lastvalues,
                    self.keep_prob: 0.5
                })  # Run training operation and compute loss, accuracy, and summaries
                
                counter += 1  # Increment counter
                idx += 1  # Increment batch index
            epochcount += 1  # Increment epoch count
            idx = 0

            self.save(self.checkpoint_dir, counter, epochcount)  # Save model checkpoint after each epoch

            acc, auc, model_name = self.test(self.test_set, epochcount)  # Test model after each epoch
            if auc > max_auc:
                max_auc = auc  # Update maximum AUC if the new AUC is higher
            result_file.write("epoch: " + str(epochcount) + "," + str(acc) + "," + str(auc) + "\r\n")  # Write results to file
            print("")

        result_file.close()
        return max_auc  # Return maximum AUC score

    # Function to test the model
    def test(self, dataset, epoch):
        start_time = time.time()  # Start the timer
        counter = 0
        dataset.shuffle(self.batch_size, False)  # Shuffle dataset for testing
        totalacc = 0.0
        totalauc = 0.0
        auccounter = 0  # Initialize AUC counter
        for data_x, data_y, data_mean, data_m, data_delta, data_x_lengths, data_lastvalues, _, _, _ in dataset.nextBatch():
            summary_str, acc, pred = self.sess.run([self.sum, self.accuracy, self.pred], feed_dict={
                self.x: data_x,
                self.y: data_y,
                self.m: data_m,
                self.delta: data_delta,
                self.mean: data_mean,
                self.x_lengths: data_x_lengths,
                self.lastvalues: data_lastvalues,
                self.keep_prob: 1.0
            })  # Run the model on test data and get accuracy and predictions
            
            try:
                auc = metrics.roc_auc_score(np.array(data_y), np.array(pred))  # Compute AUC score
                totalauc += auc  # Accumulate AUC scores
                auccounter += 1  # Increment AUC counter
                print("Batch: %4d time: %4.4f, acc: %.8f, auc: %.8f" % (counter, time.time() - start_time, acc, auc))  # Print batch results
            except ValueError:
                print("Batch: %4d time: %4.4f, acc: %.8f " % (counter, time.time() - start_time, acc))  # Print results if AUC calculation fails
                pass
            totalacc += acc  # Accumulate accuracy scores
            counter += 1  # Increment batch counter
        
        totalacc = totalacc / counter  # Compute average accuracy
        try:
            totalauc = totalauc / auccounter  # Compute average AUC
        except:
            pass
        print("epoch is : %2.2f, Total acc: %.8f, Total auc: %.8f , counter is : %.2f , auccounter is %.2f" % (epoch, totalacc, totalauc, counter, auccounter))  # Print total results for the epoch
        f = open(os.path.join(self.checkpoint_dir, self.model_dir(epoch), self.model_name, "final_acc_and_auc"), "w")  # Open file to write final results
        f.write(str(totalacc) + "," + str(totalauc))  # Write accuracy and AUC to file
        f.close()
        return totalacc, totalauc, self.model_name  # Return final accuracy, AUC, and model name
  
