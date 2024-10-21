# This script initializes gru_delta_forGAN.py to complete the evaluation of the imputed results created by WGAN_GRUI.py.
# It uses a Recurrent Nueral Network (gru_delta_forGAN.grui) that trains on the imputed training data and uses the imputed test data to measure performance.
# This is essentially the evaluation step of the Machine Learning process.

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:47:41 2018

@author: yonghong
"""

from __future__ import print_function
import sys
sys.path.append("..")                                                                                   # Adds the parent directory to the system path. Imports custom modules from a higher directory.
import argparse                                                                                         # Used to parse command line arguments and convert them to usable variables.
import os                                                                                               # Used to assist with working with files and directory paths.
import tensorflow as tf                                                                                 # Imports TensorFlow (the deep learning library) for building and training neural networks.
from Physionet2012ImputedData import readImputed                                                        ## Import readImputed.py for use with script. Why???
import gru_delta_forGAN                                                                                 ## Import gru_delta_forGAN for use.  Why???

#parse arguments
# This creates command line arguments (variables) that will be passed to the WGAN. 
# This is useful because we can change any parameter from  the command line when we run the code without having to change the code itself.
# This creates ease of use when trying to increase the performance of the WGAN by finding the right parameter values.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)                                                                # Specifies which GPU(s) to use.
    parser.add_argument('--batch-size', type=int, default=128)                                                             # Specifies the batch size for training.
    parser.add_argument('--run-type', type=str, default='test')                                                            # Specifies the type of run (e.g., test or train)
    parser.add_argument('--data-path', type=str, default="../Gan_Imputation/imputation_train_results/WGAN_no_mask/")       # Enter the complete path of the training dataset after imputation. Gan_Imputation/imputation_train_results/WGAN_no_mask/30_8_128_64_0.001_400_True_True_True_0.15_0.5
    parser.add_argument('--model-path', type=str, default=None)                                                            # Specifies the path to the model.
    parser.add_argument('--result-path', type=str, default=None)                                                           # Specifies where to save results.
    parser.add_argument('--lr', type=float, default=0.01)                                                                  # Specifies the learning rate.
    #parser.add_argument('--epoch', type=int, default=20)                                                                  
    parser.add_argument('--n-inputs', type=int, default=41)                                                                # Specifies the number of input features.
    parser.add_argument('--n-hidden-units', type=int, default=64)                                                          # Specifies the number of neurons in the hidden layer.
    parser.add_argument('--n-classes', type=int, default=2)                                                                # Specifies the number of output classes.
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint_physionet_imputed',                              # Specifies where to save model checkpoints.
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs_physionet_imputed',                                           # Specifies where to save training logs.
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)                                                                   # Determine whether to apply batch normalization.
    parser.add_argument('--isSlicing',type=int,default=1)                                                                  # Determines whether the data should be sliced into smaller sections for processing (used for performance management).
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)                                                             # Determine whether to apply batch normalization.
    args = parser.parse_args()
    
    #This converts these integer command line arguments for determing if batch normalization and slicing are required to boolean values.
    #This makes the WGAN code more readable and easier to use.
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True
            
    
    checkdir=args.checkpoint_dir                                               # Initialize the path for the checkpoints.
    logdir=args.log_dir                                                        # Initialize the path for the logs
    base=args.data_path                                                        # Initialize the base data path.
    data_paths=["30_8_128_64_0.001_400_True_True_True_0.15_0.5"]               # Data directories (it contains a single value).
    max_auc = 0.0                                                              # This variable stores the maximum AUC (Area Under the Curve) score during training.
    for d in data_paths:                                                       # Loops over each data path in data_paths.
        args.data_path=os.path.join(base,d)                                    # Append current path (d) to the base path (base).
        path_splits=args.data_path.split("/")                                  # Splits the path on "/".
        if len(path_splits[-1])==0:                                            # Lines (69 - 72) Checks if the last part of the is empty and uses the second to last part if so.  Assigns dataset name depending on condition.
            datasetName=path_splits[-2]
        else:
            datasetName=path_splits[-1]
        args.checkpoint_dir=checkdir+"/"+datasetName                                                                                           # Include dataset name in checkpoint directory.
        args.log_dir=logdir+"/"+datasetName                                                                                                    # Include dataset name in log directory.
        
        dt_train=readImputed.ReadImputedPhysionetData(args.data_path)                                                                          # Initialize ReadImputedPhysionetData to load imputed training data.
        dt_train.load()                                                                                                                        # Reads in imputed training data. 
        
        dt_test=readImputed.ReadImputedPhysionetData(args.data_path.replace("imputation_train_results","imputation_test_results"))             # Initialize ReadImputedPhysionetData to load imputed test data.
        dt_test.load()                                                                                                                         # Reads in imputed test data.
          
        lrs=[0.004,0.003,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015]                         # Initializes a list of learning rates to experiment with.
        #lrs = [0.0075,0.0085]
        for lr in lrs:                                                                           # Loop over learning rates.
            args.lr=lr                                                                           # Update learning rate to current value in loop.
            epoch=30                                                                             # Initialize number of epochs for each learning rate to 30.
            args.epoch=epoch                                                                     # Updates number epochs to 30
            print("epoch: %2d"%(epoch))                                                          # Print out current epoch that is running.
            tf.reset_default_graph()                                                             # Reset computation graph to prevent unexpected behavior and memory leaks.
            config = tf.ConfigProto()                                                            # Create a configuration object which will help guide how TF behaves.
            config.gpu_options.allow_growth = True                                               # Setting this to true will allocate GPU memory only as needed.  Will help all processes run more smoothly on the same GPU.
            with tf.Session(config=config) as sess:                                              # Create new TensorFlow session.  This session will be used to execute all the operations in the computation graph.
                model = gru_delta_forGAN.grui(sess,                                              # Create gru_delta_forGAN.grui instance with args (default and any modifications passed), TF session, and read in training set and test set.
                            args=args,
                            dataset=dt_train,
                            test_set = dt_test
                            )
            
                
                model.build()                                                                    # Build gru_delta_forGAN.grui graph.
            
                auc = model.train()                                                              # Train gru_delta_forGAN.grui graph
                if auc > max_auc:                                                                # Determines if the new auc score is higher and reassigns the value of max_auc if so.
                    max_auc = auc 
                
            print("")
        print("max auc is: " + str(max_auc))                                                     # Prints the highest auc score.
        f2 = open("max_auc","w")                                    
        f2.write(str(max_auc))                                                                   # Writes the highest AUC score to the file namesed max_auc.
        f2.close()


